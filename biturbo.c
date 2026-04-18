/*
 * biturbo.c — Zero-dependency BitNet 1.58-bit inference engine
 *             with TurboQuant INT4 KV cache quantization
 *
 * Loads GGUF models directly. Implements the full 15-stage pipeline:
 *   1.  BPE tokenization (from GGUF metadata)
 *   2.  Token embedding (F16)
 *   3.  RMS norm (pre-attention)
 *   4.  BitLinear INT8 input quantization
 *   5.  1.58-bit ternary weights (I2_S group-interleaved)
 *   6.  BitLinear GEMM (INT8 × ternary)
 *   7.  Q/K/V projection (separate matrices)
 *   8.  RoPE positional encoding
 *   9.  KV cache (TurboQuant uniform INT4)
 *   10. Attention score (GQA)
 *   11. Attention sub-norm + output projection
 *   12. FFN: pre-norm + SqReLU gate
 *   13. FFN: sub-norm + down projection
 *   14. Final norm + LM head (tied to embedding)
 *   15. Sampling → decode
 */

#include "biturbo.h"
#include "biturbo_btpk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef BT_FPGA
#include "biturbo_fpga.h"
#endif

/* ================================================================
 * VOCAB SORT HELPER (for building binary search index)
 * ================================================================ */

static char** bt_sort_vocab_global;
static int bt_sort_vocab_cmp(const void* a, const void* b) {
    return strcmp(bt_sort_vocab_global[*(const int*)a],
                 bt_sort_vocab_global[*(const int*)b]);
}

/* ================================================================
 * MEMORY HELPERS
 * ================================================================ */

static void* bt_calloc(size_t count, size_t size) {
    void* p = calloc(count, size);
    if (!p) {
        fprintf(stderr, "biturbo: OOM allocating %zu bytes\n", count * size);
        exit(1);
    }
    return p;
}

static double bt_timespec_elapsed_sec(struct timespec start,
                                      struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static void bt_profile_accum_elapsed(double *acc, struct timespec start) {
    struct timespec end;

    clock_gettime(CLOCK_MONOTONIC, &end);
    *acc += bt_timespec_elapsed_sec(start, end);
}

static double bt_profile_ms_per_layer_token(double total_sec,
                                            double layer_tokens) {
    if (layer_tokens <= 0.0)
        return 0.0;
    return total_sec * 1e3 / layer_tokens;
}

static void bt_profile_print_layer_stage_summary(const double *totals,
                                                 int gen_count,
                                                 int n_layers) {
    double layer_tokens;

    if (!totals || gen_count <= 0 || n_layers <= 0)
        return;

    layer_tokens = (double)gen_count * (double)n_layers;

    fprintf(stderr, "  layer stages (totals, ms/layer-token):\n");
    fprintf(stderr,
            "    attn_norm=%7.2fs %8.3f | wt_load=%7.2fs %8.3f | qkv=%7.2fs %8.3f\n",
            totals[BT_PROFILE_LAYER_STAGE_ATTN_NORM],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_LAYER_STAGE_ATTN_NORM], layer_tokens),
            totals[BT_PROFILE_LAYER_STAGE_WEIGHT_LOAD],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_LAYER_STAGE_WEIGHT_LOAD], layer_tokens),
            totals[BT_PROFILE_LAYER_STAGE_QKV],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_LAYER_STAGE_QKV], layer_tokens));
    fprintf(stderr,
            "    rope     =%7.2fs %8.3f | kv_cache=%7.2fs %8.3f | attention=%7.2fs %8.3f\n",
            totals[BT_PROFILE_LAYER_STAGE_ROPE],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_LAYER_STAGE_ROPE], layer_tokens),
            totals[BT_PROFILE_LAYER_STAGE_KV_CACHE],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_LAYER_STAGE_KV_CACHE], layer_tokens),
            totals[BT_PROFILE_LAYER_STAGE_ATTENTION],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_LAYER_STAGE_ATTENTION], layer_tokens));
    fprintf(stderr,
            "    attn_out =%7.2fs %8.3f | attn_res=%7.2fs %8.3f | ffn_in=%7.2fs %8.3f\n",
            totals[BT_PROFILE_LAYER_STAGE_ATTN_OUT],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_LAYER_STAGE_ATTN_OUT], layer_tokens),
            totals[BT_PROFILE_LAYER_STAGE_ATTN_RESIDUAL],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_LAYER_STAGE_ATTN_RESIDUAL], layer_tokens),
            totals[BT_PROFILE_LAYER_STAGE_FFN_IN],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_LAYER_STAGE_FFN_IN], layer_tokens));
    fprintf(stderr,
            "    sqrelu   =%7.2fs %8.3f | ffn_out=%7.2fs %8.3f | ffn_res=%7.2fs %8.3f\n",
            totals[BT_PROFILE_LAYER_STAGE_SQRELU],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_LAYER_STAGE_SQRELU], layer_tokens),
            totals[BT_PROFILE_LAYER_STAGE_FFN_OUT],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_LAYER_STAGE_FFN_OUT], layer_tokens),
            totals[BT_PROFILE_LAYER_STAGE_FFN_RESIDUAL],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_LAYER_STAGE_FFN_RESIDUAL], layer_tokens));
}

static void bt_profile_print_qkv_stage_summary(const double *totals,
                                               int gen_count,
                                               int n_layers) {
    double layer_tokens;

    if (!totals || gen_count <= 0 || n_layers <= 0)
        return;

    layer_tokens = (double)gen_count * (double)n_layers;

    fprintf(stderr, "  qkv detail (totals, ms/layer-token):\n");
    fprintf(stderr,
            "    quant    =%7.2fs %8.3f | prep =%7.2fs %8.3f | upload=%7.2fs %8.3f\n",
            totals[BT_PROFILE_QKV_STAGE_QUANT],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_QKV_STAGE_QUANT], layer_tokens),
            totals[BT_PROFILE_QKV_STAGE_PREP],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_QKV_STAGE_PREP], layer_tokens),
            totals[BT_PROFILE_QKV_STAGE_UPLOAD],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_QKV_STAGE_UPLOAD], layer_tokens));
    fprintf(stderr,
            "    wq       =%7.2fs %8.3f | wk   =%7.2fs %8.3f | wv   =%7.2fs %8.3f\n",
            totals[BT_PROFILE_QKV_STAGE_WQ],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_QKV_STAGE_WQ], layer_tokens),
            totals[BT_PROFILE_QKV_STAGE_WK],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_QKV_STAGE_WK], layer_tokens),
            totals[BT_PROFILE_QKV_STAGE_WV],
            bt_profile_ms_per_layer_token(
                totals[BT_PROFILE_QKV_STAGE_WV], layer_tokens));
}

/* ================================================================
 * §1. GGUF PARSER — reads model config, tokenizer, and weight pointers
 *
 * GGUF v3 format:
 *   [magic:u32] [version:u32] [n_tensors:u64] [n_kv:u64]
 *   [metadata kv pairs...]
 *   [tensor info headers...]
 *   [alignment padding]
 *   [tensor data...]
 * ================================================================ */

#define GGUF_MAGIC    0x46554747  /* "GGUF" */
#define GGUF_TYPE_U8  0
#define GGUF_TYPE_I8  1
#define GGUF_TYPE_U16 2
#define GGUF_TYPE_I16 3
#define GGUF_TYPE_U32 4
#define GGUF_TYPE_I32 5
#define GGUF_TYPE_F32 6
#define GGUF_TYPE_BOOL 7
#define GGUF_TYPE_STR 8
#define GGUF_TYPE_ARR 9
#define GGUF_TYPE_U64 10
#define GGUF_TYPE_I64 11
#define GGUF_TYPE_F64 12

/* GGUF tensor types */
#define GGUF_TENSOR_F32  0
#define GGUF_TENSOR_F16  1
#define GGUF_TENSOR_Q6K  14
#define GGUF_TENSOR_I2S  36

/* Q6_K block: 256 elements in ~210 bytes (6.5625 bits/weight) */
#define BT_QK_K 256

typedef struct {
    uint8_t ql[BT_QK_K/2];      /* 128 bytes: lower 4 bits of quants */
    uint8_t qh[BT_QK_K/4];      /*  64 bytes: upper 2 bits of quants */
    int8_t  scales[BT_QK_K/16]; /*  16 bytes: per-sub-block scales   */
    uint16_t d;                  /*   2 bytes: super-block scale (F16)*/
} bt_block_q6k_t;

/* Dequantize one row of Q6_K blocks into float */
static void dequantize_q6k(const bt_block_q6k_t* x, float* y, int k) {
    int nb = k / BT_QK_K;
    for (int i = 0; i < nb; i++) {
        float d = bt_f16_to_f32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  sc = x[i].scales;
        for (int n = 0; n < BT_QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l+ 0] & 0xF) | (((qh[l]>>0)&3)<<4)) - 32;
                int8_t q2 = (int8_t)((ql[l+32] & 0xF) | (((qh[l]>>2)&3)<<4)) - 32;
                int8_t q3 = (int8_t)((ql[l+ 0]  >> 4) | (((qh[l]>>4)&3)<<4)) - 32;
                int8_t q4 = (int8_t)((ql[l+32]  >> 4) | (((qh[l]>>6)&3)<<4)) - 32;
                y[l+ 0] = d * sc[is+0] * q1;
                y[l+32] = d * sc[is+2] * q2;
                y[l+64] = d * sc[is+4] * q3;
                y[l+96] = d * sc[is+6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

typedef struct {
    const uint8_t* base;
    size_t pos;
    size_t size;
} gguf_reader_t;

static uint32_t rd_u32(gguf_reader_t* r) {
    uint32_t v;
    memcpy(&v, r->base + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t rd_u64(gguf_reader_t* r) {
    uint64_t v;
    memcpy(&v, r->base + r->pos, 8);
    r->pos += 8;
    return v;
}

static float rd_f32(gguf_reader_t* r) {
    float v;
    memcpy(&v, r->base + r->pos, 4);
    r->pos += 4;
    return v;
}

static double rd_f64(gguf_reader_t* r) {
    double v;
    memcpy(&v, r->base + r->pos, 8);
    r->pos += 8;
    return v;
}

/* Read GGUF string: u64 length + bytes (NOT null-terminated) */
typedef struct { const char* str; uint64_t len; } gguf_str_t;

static gguf_str_t rd_str(gguf_reader_t* r) {
    gguf_str_t s;
    s.len = rd_u64(r);
    s.str = (const char*)(r->base + r->pos);
    r->pos += s.len;
    return s;
}

static int str_eq(gguf_str_t s, const char* lit) {
    size_t ll = strlen(lit);
    return s.len == ll && memcmp(s.str, lit, ll) == 0;
}

/* Skip a value of given type */
static void skip_val(gguf_reader_t* r, uint32_t type);

static size_t type_size(uint32_t t) {
    switch (t) {
        case GGUF_TYPE_U8: case GGUF_TYPE_I8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_U16: case GGUF_TYPE_I16: return 2;
        case GGUF_TYPE_U32: case GGUF_TYPE_I32: case GGUF_TYPE_F32: return 4;
        case GGUF_TYPE_U64: case GGUF_TYPE_I64: case GGUF_TYPE_F64: return 8;
        default: return 0;
    }
}

static void skip_val(gguf_reader_t* r, uint32_t type) {
    if (type == GGUF_TYPE_STR) {
        rd_str(r);
    } else if (type == GGUF_TYPE_ARR) {
        uint32_t etype = rd_u32(r);
        uint64_t count = rd_u64(r);
        if (etype == GGUF_TYPE_STR) {
            for (uint64_t i = 0; i < count; i++) rd_str(r);
        } else {
            r->pos += count * type_size(etype);
        }
    } else {
        r->pos += type_size(type);
    }
}

/* Tensor info from GGUF header */
typedef struct {
    const char* name;
    int name_len;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;  /* relative to data section start */
} gguf_tensor_info_t;

/* Find a tensor by name */
static const gguf_tensor_info_t* find_tensor(const gguf_tensor_info_t* tensors,
                                              int n, const char* name) {
    int nl = (int)strlen(name);
    for (int i = 0; i < n; i++) {
        if (tensors[i].name_len == nl &&
            memcmp(tensors[i].name, name, nl) == 0)
            return &tensors[i];
    }
    return NULL;
}

/* ================================================================
 * §2. RMS NORM
 * ================================================================ */

static void rms_norm(float* out, const float* x, const float* weight,
                     int size, float eps) {
    float ss = 0.0f;
#ifdef __ARM_NEON
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
#ifdef __aarch64__
        acc = vfmaq_f32(acc, v, v);
#else
        acc = vmlaq_f32(acc, v, v);
#endif
    }
#ifdef __aarch64__
    ss = vaddvq_f32(acc);
#else
    float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    sum2 = vpadd_f32(sum2, sum2);
    ss = vget_lane_f32(sum2, 0);
#endif
    for (; i < size; i++) ss += x[i] * x[i];
#else
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
#endif
    float scale = 1.0f / sqrtf(ss / (float)size + eps);
#ifdef __ARM_NEON
    {
        float32x4_t vs = vdupq_n_f32(scale);
        int i = 0;
        for (; i + 3 < size; i += 4) {
            float32x4_t vx = vld1q_f32(x + i);
            float32x4_t vw = vld1q_f32(weight + i);
            vst1q_f32(out + i, vmulq_f32(vmulq_f32(vx, vs), vw));
        }
        for (; i < size; i++) out[i] = x[i] * scale * weight[i];
    }
#else
    for (int i = 0; i < size; i++) out[i] = x[i] * scale * weight[i];
#endif
}

/* ================================================================
 * §3. BITLINEAR INPUT QUANTIZATION (Stage 4)
 *
 * Per-token: scale = 127 / max(|x|), x_q = round(clamp(x * scale))
 * Returns inverse scale for post-GEMM dequantization.
 * ================================================================ */

static float bitlinear_quantize(int8_t* out, const float* x, int size) {
    float amax = 0.0f;
#ifdef __ARM_NEON
    {
        float32x4_t vmax = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 3 < size; i += 4) {
            float32x4_t v = vld1q_f32(x + i);
            vmax = vmaxq_f32(vmax, vabsq_f32(v));
        }
        /* Horizontal max */
        float32x2_t m2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        m2 = vpmax_f32(m2, m2);
        amax = vget_lane_f32(m2, 0);
        for (; i < size; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
    }
#else
    for (int i = 0; i < size; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
#endif
    if (amax < 1e-10f) amax = 1e-10f;
    float scale = 127.0f / amax;
#ifdef __ARM_NEON
    {
        float32x4_t vscale = vdupq_n_f32(scale);
        float32x4_t vhalf  = vdupq_n_f32(0.5f);
        float32x4_t vnhalf = vdupq_n_f32(-0.5f);
        int32x4_t   vmax_i = vdupq_n_s32(127);
        int32x4_t   vmin_i = vdupq_n_s32(-128);
        int i = 0;
        for (; i + 7 < size; i += 8) {
            /* Process 8 elements → 8 int8 outputs */
            float32x4_t v0 = vmulq_f32(vld1q_f32(x + i), vscale);
            float32x4_t v1 = vmulq_f32(vld1q_f32(x + i + 4), vscale);
            /* Round: add ±0.5 then truncate */
            float32x4_t r0 = vaddq_f32(v0, vbslq_f32(vcltq_f32(v0, vdupq_n_f32(0.0f)),
                                        vnhalf, vhalf));
            float32x4_t r1 = vaddq_f32(v1, vbslq_f32(vcltq_f32(v1, vdupq_n_f32(0.0f)),
                                        vnhalf, vhalf));
            int32x4_t i0 = vcvtq_s32_f32(r0);
            int32x4_t i1 = vcvtq_s32_f32(r1);
            /* Clamp to [-128, 127] */
            i0 = vminq_s32(vmaxq_s32(i0, vmin_i), vmax_i);
            i1 = vminq_s32(vmaxq_s32(i1, vmin_i), vmax_i);
            /* Narrow: 32→16→8 */
            int16x4_t s0 = vmovn_s32(i0);
            int16x4_t s1 = vmovn_s32(i1);
            int8x8_t  b  = vmovn_s16(vcombine_s16(s0, s1));
            vst1_s8(out + i, b);
        }
        for (; i < size; i++) {
            int v = (int)roundf(x[i] * scale);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            out[i] = (int8_t)v;
        }
    }
#else
    for (int i = 0; i < size; i++) {
        int v = (int)roundf(x[i] * scale);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        out[i] = (int8_t)v;
    }
#endif
    return amax / 127.0f;  /* inv_scale */
}

/* ================================================================
 * §4. I2_S TERNARY GEMM (Stages 5-6)
 *
 * Group-interleaved 2-bit packing (blocks of 128 elements, 32 bytes):
 *   byte[gp] (gp=0..31):
 *     bits 7-6 → element at gp +  0  (group 0)
 *     bits 5-4 → element at gp + 32  (group 1)
 *     bits 3-2 → element at gp + 64  (group 2)
 *     bits 1-0 → element at gp + 96  (group 3)
 *   2-bit code: 0=-1, 1=0, 2=+1
 *
 * For each output row: sum ternary_weight[col] * int8_input[col]
 * ================================================================ */

static const int8_t I2S_MAP[4] = {-1, 0, 1, 0};

static void i2s_gemv(float* out, const bt_i2s_weight_t* w,
                     const int8_t* x_q, float inv_scale) {
    int rows = w->rows;
    int cols = w->cols;
    float w_scale = w->scale;
    int n_blocks = (cols + 127) / 128;     /* blocks of 128 elements per row */
    int bytes_per_row = n_blocks * 32;     /* 32 bytes per block */

    for (int r = 0; r < rows; r++) {
        const uint8_t* row_data = w->data + (size_t)r * bytes_per_row;
        int32_t acc = 0;
        int col = 0;

        for (int blk = 0; blk < n_blocks; blk++) {
            const uint8_t* bp = row_data + blk * 32;
            int blk_elems = cols - col;
            if (blk_elems > 128) blk_elems = 128;

            int cols0 = blk_elems >= 32  ? 32 : blk_elems;
            int cols1 = blk_elems >= 64  ? 32 : (blk_elems > 32  ? blk_elems - 32  : 0);
            int cols2 = blk_elems >= 96  ? 32 : (blk_elems > 64  ? blk_elems - 64  : 0);
            int cols3 = blk_elems >= 128 ? 32 : (blk_elems > 96  ? blk_elems - 96  : 0);

            for (int gp = 0; gp < 32; gp++) {
                uint8_t b = bp[gp];
                int8_t c0 = I2S_MAP[(b >> 6) & 3];
                int8_t c1 = I2S_MAP[(b >> 4) & 3];
                int8_t c2 = I2S_MAP[(b >> 2) & 3];
                int8_t c3 = I2S_MAP[(b >> 0) & 3];

                if (gp < cols0) acc += (int32_t)c0 * (int32_t)x_q[col + 0*32 + gp];
                if (gp < cols1) acc += (int32_t)c1 * (int32_t)x_q[col + 1*32 + gp];
                if (gp < cols2) acc += (int32_t)c2 * (int32_t)x_q[col + 2*32 + gp];
                if (gp < cols3) acc += (int32_t)c3 * (int32_t)x_q[col + 3*32 + gp];
            }
            col += blk_elems;
        }
        out[r] = (float)acc * inv_scale * w_scale;
    }
}

/* ================================================================
 * §4b. T-MAC TL2 LOOKUP TABLE GEMV
 *
 * Packs 3 ternary weights → 4-bit LUT index + 1-bit sign.
 * LUT has 16 int16 entries (14 used) precomputed from each
 * activation triple. GEMV becomes table lookup + accumulate.
 *
 * Encoding: canonical form has first nonzero weight = +1.
 *   If first nonzero is -1, negate all three weights, sign = 1.
 *   nibble indexes the unsigned pattern, sign negates the result.
 * ================================================================ */

/*
 * Three-weight encoding table: TMAC3_ENC[w0+1][w1+1][w2+1]
 * Value = (nibble_index << 1) | sign_bit
 *
 * LUT index → unsigned coefficient pattern (c0, c1, c2):
 *  0: (0,0,0)      1: (0,0,+1)    2: (0,+1,0)     3: (+1,0,0)
 *  4: (0,+1,+1)    5: (+1,0,+1)   6: (+1,+1,0)    7: (0,+1,-1)
 *  8: (+1,0,-1)    9: (+1,-1,0)  10: (+1,+1,+1)  11: (+1,+1,-1)
 * 12: (+1,-1,+1)  13: (+1,-1,-1)  14: unused       15: unused
 */
static const uint8_t TMAC3_ENC[3][3][3] = {
    /* w0 = -1 (index 0) */
    {   /* w1 = -1 */
        { (10<<1)|1, ( 6<<1)|1, (11<<1)|1 },   /* w2 = -1, 0, +1 */
        /* w1 = 0 */
        { ( 5<<1)|1, ( 3<<1)|1, ( 8<<1)|1 },
        /* w1 = +1 */
        { (12<<1)|1, ( 9<<1)|1, (13<<1)|1 }
    },
    /* w0 = 0 (index 1) */
    {   /* w1 = -1 */
        { ( 4<<1)|1, ( 2<<1)|1, ( 7<<1)|1 },
        /* w1 = 0 */
        { ( 1<<1)|1, ( 0<<1)|0, ( 1<<1)|0 },
        /* w1 = +1 */
        { ( 7<<1)|0, ( 2<<1)|0, ( 4<<1)|0 }
    },
    /* w0 = +1 (index 2) */
    {   /* w1 = -1 */
        { (13<<1)|0, ( 9<<1)|0, (12<<1)|0 },
        /* w1 = 0 */
        { ( 8<<1)|0, ( 3<<1)|0, ( 5<<1)|0 },
        /* w1 = +1 */
        { (11<<1)|0, ( 6<<1)|0, (10<<1)|0 }
    }
};

/* Two-weight encoding: TMAC2_ENC[w0+1][w1+1] = nibble index (no sign) */
static const uint8_t TMAC2_ENC[3][3] = {
    /* w0=-1 */ { 8, 6, 7 },   /* w1 = -1, 0, +1 */
    /* w0= 0 */ { 2, 0, 1 },
    /* w0=+1 */ { 5, 3, 4 }
};

/* Decode a single ternary value from I2_S weight at (row, col) */
static inline int8_t i2s_decode(const bt_i2s_weight_t* w, int row, int col) {
    int bytes_per_row = ((w->cols + 127) / 128) * 32;
    int block = col / 128;
    int group = (col % 128) / 32;
    int pos   = col % 32;
    uint8_t byte = w->data[(size_t)row * bytes_per_row + block * 32 + pos];
    return I2S_MAP[(byte >> (6 - 2 * group)) & 3];
}

/* Repack I2_S weight into T-MAC TL2 format */
static void tmac_repack(bt_i2s_weight_t* w) {
    int rows = w->rows, cols = w->cols;
    int n3 = cols / 3;
    int n2 = (cols % 3 != 0) ? 1 : 0;

    bt_tmac_weight_t* tw = (bt_tmac_weight_t*)bt_calloc(1, sizeof(*tw));
    tw->rows = rows;
    tw->cols = cols;
    tw->scale = w->scale;
    tw->n3 = n3;
    tw->n2 = n2;
    tw->nib3_stride = (n3 + 1) / 2;        /* ceil(n3/2) bytes per row */
    tw->sign_stride = (n3 + 7) / 8;        /* ceil(n3/8) bytes per row */
    tw->nib2_stride = (n2 + 1) / 2;        /* 0 or 1 */

    tw->three_nib  = (uint8_t*)bt_calloc((size_t)rows * tw->nib3_stride, 1);
    tw->three_sign = (uint8_t*)bt_calloc((size_t)rows * tw->sign_stride, 1);
    if (n2 > 0)
        tw->two_nib = (uint8_t*)bt_calloc((size_t)rows * tw->nib2_stride, 1);
    else
        tw->two_nib = NULL;

    for (int r = 0; r < rows; r++) {
        /* Three-weight groups */
        for (int g = 0; g < n3; g++) {
            int8_t t0 = i2s_decode(w, r, g * 3 + 0);
            int8_t t1 = i2s_decode(w, r, g * 3 + 1);
            int8_t t2 = i2s_decode(w, r, g * 3 + 2);

            uint8_t enc = TMAC3_ENC[t0 + 1][t1 + 1][t2 + 1];
            uint8_t nib  = enc >> 1;
            uint8_t sign = enc & 1;

            /* Pack nibble: even g → high nibble, odd g → low nibble */
            size_t nib_idx = (size_t)r * tw->nib3_stride + g / 2;
            if (g & 1)
                tw->three_nib[nib_idx] |= nib;
            else
                tw->three_nib[nib_idx] |= (nib << 4);

            /* Pack sign bit */
            size_t sign_idx = (size_t)r * tw->sign_stride + g / 8;
            tw->three_sign[sign_idx] |= (sign << (g & 7));
        }

        /* Two-weight groups (remainder) */
        if (n2 > 0) {
            int col_base = n3 * 3;
            int8_t t0 = i2s_decode(w, r, col_base);
            int8_t t1 = (col_base + 1 < cols) ? i2s_decode(w, r, col_base + 1) : 0;

            uint8_t nib = TMAC2_ENC[t0 + 1][t1 + 1];
            /* Single group → always high nibble of byte 0 */
            tw->two_nib[(size_t)r * tw->nib2_stride] = (nib << 4);
        }
    }

    w->tmac = tw;
}

/* Build three-weight LUT from quantized activations */
static void tmac_build_three_lut(int16_t* lut, const int8_t* x_q, int n3) {
    for (int g = 0; g < n3; g++) {
        int16_t a0 = x_q[g * 3 + 0];
        int16_t a1 = x_q[g * 3 + 1];
        int16_t a2 = x_q[g * 3 + 2];
        int16_t* t = lut + g * 16;
        t[ 0] = 0;
        t[ 1] = a2;
        t[ 2] = a1;
        t[ 3] = a0;
        t[ 4] = a1 + a2;
        t[ 5] = a0 + a2;
        t[ 6] = a0 + a1;
        t[ 7] = a1 - a2;
        t[ 8] = a0 - a2;
        t[ 9] = a0 - a1;
        t[10] = a0 + a1 + a2;
        t[11] = a0 + a1 - a2;
        t[12] = a0 - a1 + a2;
        t[13] = a0 - a1 - a2;
        t[14] = 0;
        t[15] = 0;
    }
}

/* Build two-weight LUT from quantized activations */
static void tmac_build_two_lut(int16_t* lut, const int8_t* x_q,
                               int col_base, int cols, int n2) {
    for (int g = 0; g < n2; g++) {
        int16_t a0 = x_q[col_base + g * 2 + 0];
        int16_t a1 = (col_base + g * 2 + 1 < cols) ? x_q[col_base + g * 2 + 1] : 0;
        int16_t* t = lut + g * 16;
        t[0] =  0;       t[1] =  a1;      t[2] = -a1;
        t[3] =  a0;      t[4] =  a0 + a1; t[5] =  a0 - a1;
        t[6] = -a0;      t[7] = -a0 + a1; t[8] = -a0 - a1;
        t[9] = 0; t[10] = 0; t[11] = 0;
        t[12] = 0; t[13] = 0; t[14] = 0; t[15] = 0;
    }
}

/* T-MAC TL2 GEMV kernel (raw accumulator output) */
static void tmac_gemv_raw(int32_t* out, const bt_tmac_weight_t* tw,
                          const int16_t* three_lut, const int16_t* two_lut) {
    int rows = tw->rows;
    int n3 = tw->n3;
    int n2 = tw->n2;

    for (int r = 0; r < rows; r++) {
        int32_t acc = 0;

        /* Three-weight groups */
        const uint8_t* nib_row  = tw->three_nib  + (size_t)r * tw->nib3_stride;
        const uint8_t* sign_row = tw->three_sign + (size_t)r * tw->sign_stride;

        for (int g = 0; g < n3; g++) {
            /* Extract 4-bit nibble */
            uint8_t packed = nib_row[g / 2];
            int nibble = (g & 1) ? (packed & 0x0F) : (packed >> 4);

            /* Extract sign bit */
            int sign_bit = (sign_row[g / 8] >> (g & 7)) & 1;

            /* Lookup and accumulate */
            int16_t val = three_lut[g * 16 + nibble];
            acc += sign_bit ? -(int32_t)val : (int32_t)val;
        }

        /* Two-weight groups */
        if (n2 > 0) {
            const uint8_t* nib2_row = tw->two_nib + (size_t)r * tw->nib2_stride;
            for (int g = 0; g < n2; g++) {
                uint8_t packed = nib2_row[g / 2];
                int nibble = (g & 1) ? (packed & 0x0F) : (packed >> 4);
                acc += (int32_t)two_lut[g * 16 + nibble];
            }
        }

        out[r] = acc;
    }
}

/* T-MAC TL2 GEMV kernel */
static void tmac_gemv(float* out, const bt_tmac_weight_t* tw,
                      float inv_scale,
                      const int16_t* three_lut, const int16_t* two_lut) {
    float dequant = inv_scale * tw->scale;
    for (int r = 0; r < tw->rows; r++) {
        int32_t acc = 0;

        /* Three-weight groups */
        const uint8_t* nib_row  = tw->three_nib  + (size_t)r * tw->nib3_stride;
        const uint8_t* sign_row = tw->three_sign + (size_t)r * tw->sign_stride;

        for (int g = 0; g < tw->n3; g++) {
            uint8_t packed = nib_row[g / 2];
            int nibble = (g & 1) ? (packed & 0x0F) : (packed >> 4);
            int sign_bit = (sign_row[g / 8] >> (g & 7)) & 1;
            int16_t val = three_lut[g * 16 + nibble];
            acc += sign_bit ? -(int32_t)val : (int32_t)val;
        }

        if (tw->n2 > 0) {
            const uint8_t* nib2_row = tw->two_nib + (size_t)r * tw->nib2_stride;
            for (int g = 0; g < tw->n2; g++) {
                uint8_t packed = nib2_row[g / 2];
                int nibble = (g & 1) ? (packed & 0x0F) : (packed >> 4);
                acc += (int32_t)two_lut[g * 16 + nibble];
            }
        }

        out[r] = (float)acc * dequant;
    }
}

#ifdef BT_FPGA
static int bt_fpga_compare_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* s = getenv("BT_FPGA_COMPARE");
        cached = (s && s[0] != '\0' && strcmp(s, "0") != 0) ? 1 : 0;
    }
    return cached;
}

static void bt_fpga_compare_tmac(const char* stage, int layer,
                                 const bt_tmac_weight_t* tw,
                                 const int8_t* q8_buf, float inv_scale,
                                 const float* fpga_out) {
    static int compare_budget = 0;
    if (!bt_fpga_compare_enabled() || compare_budget >= 8)
        return;

    size_t lut_elems = (size_t)(tw->n3 + tw->n2) * 16;
    int16_t* lut_buf = (int16_t*)malloc(lut_elems * sizeof(*lut_buf));
    int32_t* raw_cpu = (int32_t*)malloc((size_t)tw->rows * sizeof(*raw_cpu));
    if (!lut_buf || !raw_cpu) {
        fprintf(stderr, "[FPGA-CMP] OOM while comparing %s L%d\n", stage, layer);
        free(lut_buf);
        free(raw_cpu);
        compare_budget++;
        return;
    }

    tmac_build_three_lut(lut_buf, q8_buf, tw->n3);
    {
        int16_t* two_lut = lut_buf + tw->n3 * 16;
        tmac_build_two_lut(two_lut, q8_buf, tw->n3 * 3, tw->cols, tw->n2);
        tmac_gemv_raw(raw_cpu, tw, lut_buf, two_lut);
    }

    float dequant = inv_scale * tw->scale;
    int worst_row = 0;
    double max_abs = 0.0;
    double sum_abs = 0.0;
    for (int r = 0; r < tw->rows; r++) {
        double cpu = (double)raw_cpu[r] * (double)dequant;
        double diff = fabs(cpu - (double)fpga_out[r]);
        sum_abs += diff;
        if (diff > max_abs) {
            max_abs = diff;
            worst_row = r;
        }
    }

    fprintf(stderr,
            "[FPGA-CMP] L%d %s rows=%d cols=%d max_abs=%.6g mean_abs=%.6g worst_row=%d\n",
            layer, stage, tw->rows, tw->cols,
            max_abs, sum_abs / (double)tw->rows, worst_row);

    {
        int sample = tw->rows < 8 ? tw->rows : 8;
        for (int i = 0; i < sample; i++) {
            float cpu = (float)raw_cpu[i] * dequant;
            fprintf(stderr,
                    "  [FPGA-CMP] row %d cpu=% .6f fpga=% .6f diff=% .6f\n",
                    i, cpu, fpga_out[i], cpu - fpga_out[i]);
        }
    }

    free(lut_buf);
    free(raw_cpu);
    compare_budget++;
}
#endif

/* T-MAC BitLinear forward: quantize → build LUTs → T-MAC GEMV */
static void tmac_forward(float* out, const float* x, int8_t* q8_buf,
                         int16_t* lut_buf, const bt_i2s_weight_t* w) {
    bt_tmac_weight_t* tw = w->tmac;
    float inv_scale = bitlinear_quantize(q8_buf, x, w->cols);
    tmac_build_three_lut(lut_buf, q8_buf, tw->n3);
    int16_t* two_lut = lut_buf + tw->n3 * 16;
    tmac_build_two_lut(two_lut, q8_buf, tw->n3 * 3, tw->cols, tw->n2);
    tmac_gemv(out, tw, inv_scale, lut_buf, two_lut);
}

/* Free T-MAC repacked weight */
static void tmac_free(bt_i2s_weight_t* w) {
    if (w->tmac) {
        bt_tmac_weight_t* tw = w->tmac;
        free(tw->three_nib);
        free(tw->three_sign);
        free(tw->two_nib);
        free(tw);
        w->tmac = NULL;
    }
}

/* ================================================================
 * §5. ROPE (Stage 8)
 * ================================================================ */

static void rope(float* vec, int head_dim, int n_heads, int pos, float theta) {
    for (int h = 0; h < n_heads; h++) {
        float* hv = vec + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float v0 = hv[i], v1 = hv[i + 1];
            hv[i]     = v0 * cos_a - v1 * sin_a;
            hv[i + 1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

/* ================================================================
 * §6. TURBOQUANT KV CACHE (Stage 9)
 *
 * Real TurboQuant: RHT + Lloyd-Max 3-bit codebook + QJL residual.
 * K attention: two-stage inner product (codebook dot + QJL XNOR)
 * V dequant:   MSE-only point-wise (codebook → inverse RHT)
 * ================================================================ */

#define TQ_DEFAULT_SEED 0x12345678u
#define TQ_PI_2         1.5707963267948966f  /* pi/2 */

/* Lloyd-Max optimal centroids for N(0,1), 3-bit (8 levels) */
static const float TQ_CENTROIDS[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1520f
};

/* ── RHT support ── */

static int tq_random_sign(uint32_t seed, int idx) {
    uint32_t h = seed ^ (uint32_t)idx;
    h = h * 2654435761u;
    return (h & 1) ? 1 : -1;
}

static void tq_walsh_hadamard(float* data, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
#ifdef __ARM_NEON
            int j = 0;
            for (; j + 3 < len; j += 4) {
                float32x4_t u = vld1q_f32(data + i + j);
                float32x4_t v = vld1q_f32(data + i + j + len);
                vst1q_f32(data + i + j,       vaddq_f32(u, v));
                vst1q_f32(data + i + j + len, vsubq_f32(u, v));
            }
            for (; j < len; j++) {
#else
            for (int j = 0; j < len; j++) {
#endif
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
}

static void tq_rht_forward(float* data, int n, uint32_t seed) {
    int n2 = 1;
    while (n2 * 2 <= n) n2 *= 2;
    for (int i = 0; i < n2; i++)
        data[i] *= (float)tq_random_sign(seed, i);
    tq_walsh_hadamard(data, n2);
    float scale = 1.0f / sqrtf((float)n2);
    for (int i = 0; i < n2; i++)
        data[i] *= scale;
}

static void tq_rht_inverse(float* data, int n, uint32_t seed) {
    int n2 = 1;
    while (n2 * 2 <= n) n2 *= 2;
    float scale = 1.0f / sqrtf((float)n2);
    for (int i = 0; i < n2; i++)
        data[i] *= scale;
    tq_walsh_hadamard(data, n2);
    for (int i = 0; i < n2; i++)
        data[i] *= (float)tq_random_sign(seed, i);
}

/* ── Codebook quantize/dequantize ── */

static void tq_codebook_quantize(const float* src, uint8_t* indices,
                                  int n, float inv_std) {
    for (int i = 0; i < n; i++) {
        float x = src[i] * inv_std;
        int best = 0;
        float best_dist = fabsf(x - TQ_CENTROIDS[0]);
        for (int c = 1; c < 8; c++) {
            float dist = fabsf(x - TQ_CENTROIDS[c]);
            if (dist < best_dist) { best_dist = dist; best = c; }
        }
        indices[i] = (uint8_t)best;
    }
}

static void tq_codebook_dequantize(const uint8_t* indices, float* dst,
                                    int n, float inv_std) {
    float std_val = (inv_std > 1e-10f) ? (1.0f / inv_std) : 1.0f;
    for (int i = 0; i < n; i++)
        dst[i] = TQ_CENTROIDS[indices[i]] * std_val;
}

/* ── 3-bit packing (LSB-first bitstream) ── */

static void tq_pack_3bit(const uint8_t* indices, uint8_t* packed, int n) {
    int total_bytes = (n * 3 + 7) / 8;
    memset(packed, 0, (size_t)total_bytes);
    for (int i = 0; i < n; i++) {
        int bit_offset = i * 3;
        int byte_idx = bit_offset / 8;
        int bit_pos  = bit_offset % 8;
        uint16_t val = (uint16_t)(indices[i] & 0x07);
        packed[byte_idx] |= (uint8_t)(val << bit_pos);
        if (bit_pos > 5)
            packed[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_pos));
    }
}

static void tq_unpack_3bit(const uint8_t* packed, uint8_t* indices, int n) {
    for (int i = 0; i < n; i++) {
        int bit_offset = i * 3;
        int byte_idx = bit_offset / 8;
        int bit_pos  = bit_offset % 8;
        uint16_t val = (uint16_t)packed[byte_idx];
        if (bit_pos > 5 && byte_idx + 1 < (n * 3 + 7) / 8)
            val |= (uint16_t)packed[byte_idx + 1] << 8;
        indices[i] = (uint8_t)((val >> bit_pos) & 0x07);
    }
}

/* ── QJL (Quantized Johnson-Lindenstrauss) ── */

static float tq_qjl_entry(int dim_idx, int sketch_idx) {
    uint32_t h = (uint32_t)(dim_idx * 2654435761u + sketch_idx * 340573321u);
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return (h & 1) ? 1.0f : -1.0f;
}

static void tq_compute_qjl_signs(const float* residual, uint8_t* signs,
                                  int dim, int n_sketch) {
    int hash_bytes = n_sketch / 8;
    memset(signs, 0, (size_t)hash_bytes);
    for (int s = 0; s < n_sketch; s++) {
        float proj = 0.0f;
        for (int d = 0; d < dim; d++)
            proj += residual[d] * tq_qjl_entry(d, s);
        if (proj > 0.0f)
            signs[s / 8] |= (uint8_t)(1 << (s % 8));
    }
}

/* ── Main TurboQuant functions ── */

/* Quantize a float vector into a TurboQuant 4-bit block.
 * Used for both K and V cache storage. */
static void tq_quantize(bt_kv_block_t* blk, const float* src, int dim) {
    /* Step 1: L2 norm */
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) norm_sq += src[i] * src[i];
    float norm = sqrtf(norm_sq);
    blk->norm = bt_f32_to_f16(norm);

    /* Step 2: Normalize to unit vector */
    float rotated[BT_QK];
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < dim; i++) rotated[i] = src[i] * inv_norm;
    for (int i = dim; i < BT_QK; i++) rotated[i] = 0.0f;

    /* Step 3: RHT to decorrelate channels */
    uint32_t seed = TQ_DEFAULT_SEED;
    blk->rht_seed = seed;
    tq_rht_forward(rotated, dim, seed);

    /* Step 4: Quantize with 3-bit Lloyd-Max codebook.
     * After RHT, coords ~ N(0, 1/sqrt(dim)), so inv_std = sqrt(dim) → N(0,1) */
    float inv_std = sqrtf((float)dim);
    uint8_t indices[BT_QK];
    tq_codebook_quantize(rotated, indices, dim, inv_std);
    tq_pack_3bit(indices, blk->mse_indices, dim);

    /* Step 5: Compute residual for QJL */
    float reconstructed[BT_QK];
    tq_codebook_dequantize(indices, reconstructed, dim, inv_std);
    float residual[BT_QK];
    float r_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        residual[i] = rotated[i] - reconstructed[i];
        r_norm_sq += residual[i] * residual[i];
    }
    for (int i = dim; i < BT_QK; i++) residual[i] = 0.0f;
    blk->residual_norm = bt_f32_to_f16(sqrtf(r_norm_sq));

    /* Step 6: QJL 1-bit sign hash on residual */
    tq_compute_qjl_signs(residual, blk->qjl_signs, dim, dim);
}

/* Two-stage attention dot product: MSE codebook + QJL correction.
 * q_rht and q_qjl are pre-computed once per query head. */
static float tq_attention_dot(const float* q_rht, const float* q_qjl,
                               const bt_kv_block_t* blk, int dim) {
    float norm = bt_f16_to_f32(blk->norm);
    float r_norm = bt_f16_to_f32(blk->residual_norm);
    float inv_std = sqrtf((float)dim);

    /* Stage 1: MSE dot in rotated space */
    uint8_t indices[BT_QK];
    tq_unpack_3bit(blk->mse_indices, indices, dim);
    float mse_dot = 0.0f;
    float std_val = 1.0f / inv_std;
    for (int d = 0; d < dim; d++)
        mse_dot += q_rht[d] * TQ_CENTROIDS[indices[d]] * std_val;

    /* Stage 2: QJL residual correction */
    float qjl_correction = 0.0f;
    int sketch_dim = dim;
    for (int s = 0; s < sketch_dim; s++) {
        int bit = (blk->qjl_signs[s / 8] >> (s % 8)) & 1;
        float key_sign = bit ? 1.0f : -1.0f;
        qjl_correction += q_qjl[s] * key_sign;
    }
    qjl_correction *= sqrtf(TQ_PI_2) / (float)sketch_dim * r_norm;

    return norm * (mse_dot + qjl_correction);
}

/* MSE-only point-wise dequant + weighted accumulate (for V cache).
 * No QJL — QJL is only for inner product estimation. */
static void tq_dequant_accum(float* out, float weight,
                              const bt_kv_block_t* blk, int dim) {
    float norm = bt_f16_to_f32(blk->norm);
    float inv_std = sqrtf((float)dim);

    /* Dequantize in rotated space */
    uint8_t indices[BT_QK];
    tq_unpack_3bit(blk->mse_indices, indices, dim);
    float rotated[BT_QK];
    tq_codebook_dequantize(indices, rotated, dim, inv_std);

    /* Inverse RHT to return to original space */
    tq_rht_inverse(rotated, dim, blk->rht_seed);

    /* Scale by norm and accumulate */
    float w = weight * norm;
    for (int i = 0; i < dim; i++)
        out[i] += w * rotated[i];
}

/* ================================================================
 * §7. SOFTMAX
 * ================================================================ */

static void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < size; i++) x[i] *= inv;
}

/* ================================================================
 * §8. FORWARD PASS (Stages 2-14)
 * ================================================================ */

void bt_forward(bt_model_t* model, int token, int pos) {
    bt_config_t* cfg = &model->config;
    bt_weights_t* w = &model->weights;
    bt_state_t* s = &model->state;
    struct timespec ts_layers_start, ts_layers_end;
    struct timespec ts_lm_head_start, ts_lm_head_end;
    struct timespec ts_stage_start;
    struct timespec ts_sub_start;

    int dim = cfg->dim;
    int head_dim = BT_HEAD_DIM(cfg);
    int n_heads = cfg->n_heads;
    int n_kv_heads = cfg->n_kv_heads;
    int gqa_groups = BT_GQA_GROUPS(cfg);
    int ffn_dim = cfg->ffn_dim;

    /* ── Stage 2: Token embedding (Q6_K → F32) ── */
    {
        int nb_per_row = dim / BT_QK_K;
        size_t row_bytes = (size_t)nb_per_row * sizeof(bt_block_q6k_t);
        const bt_block_q6k_t* emb = (const bt_block_q6k_t*)(w->token_embedding + (size_t)token * row_bytes);
        dequantize_q6k(emb, s->x, dim);
    }

    /* ── Transformer layers ── */
    memset(s->profile_last_layer_stage_sec, 0,
           sizeof(s->profile_last_layer_stage_sec));
    memset(s->profile_last_qkv_stage_sec, 0,
           sizeof(s->profile_last_qkv_stage_sec));
    clock_gettime(CLOCK_MONOTONIC, &ts_layers_start);
    for (int l = 0; l < cfg->n_layers; l++) {
        bt_layer_weights_t* lw = &w->layers[l];
        bt_kv_cache_t* kv = &s->kv[l];

        /* Stage 3: Pre-attention RMS norm */
        clock_gettime(CLOCK_MONOTONIC, &ts_stage_start);
        rms_norm(s->xb, s->x, lw->attn_norm, dim, cfg->norm_eps);
        bt_profile_accum_elapsed(
            &s->profile_last_layer_stage_sec[BT_PROFILE_LAYER_STAGE_ATTN_NORM],
            ts_stage_start);

#ifdef BT_FPGA
        if (bt_fpga_regs) {
            clock_gettime(CLOCK_MONOTONIC, &ts_stage_start);
            bt_fpga_load_layer(model, l);
            bt_profile_accum_elapsed(
                &s->profile_last_layer_stage_sec[BT_PROFILE_LAYER_STAGE_WEIGHT_LOAD],
                ts_stage_start);
        }
#endif

        /* Stage 4-6-7: Q/K/V projections (BitLinear + T-MAC) */
        clock_gettime(CLOCK_MONOTONIC, &ts_stage_start);
        clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
        float inv_scale = bitlinear_quantize(s->q8_buf, s->xb, dim);
        bt_profile_accum_elapsed(
            &s->profile_last_qkv_stage_sec[BT_PROFILE_QKV_STAGE_QUANT],
            ts_sub_start);
#ifdef BT_FPGA
        if (bt_fpga_regs) {
            clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
            bt_fpga_upload_activations(s->q8_buf, dim);
            bt_profile_accum_elapsed(
                &s->profile_last_qkv_stage_sec[BT_PROFILE_QKV_STAGE_UPLOAD],
                ts_sub_start);

            if (model->btpk) {
                btpk_layer_t* bl = &model->btpk->layers[l];
                clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
                bt_fpga_gemv_dequant_btpk(s->q, &bl->wq, &bt_fpga_layer_locs[l].wq, inv_scale);
                bt_profile_accum_elapsed(
                    &s->profile_last_qkv_stage_sec[BT_PROFILE_QKV_STAGE_WQ],
                    ts_sub_start);
                clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
                bt_fpga_gemv_dequant_btpk(s->k, &bl->wk, &bt_fpga_layer_locs[l].wk, inv_scale);
                bt_profile_accum_elapsed(
                    &s->profile_last_qkv_stage_sec[BT_PROFILE_QKV_STAGE_WK],
                    ts_sub_start);
                clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
                bt_fpga_gemv_dequant_btpk(s->v, &bl->wv, &bt_fpga_layer_locs[l].wv, inv_scale);
                bt_profile_accum_elapsed(
                    &s->profile_last_qkv_stage_sec[BT_PROFILE_QKV_STAGE_WV],
                    ts_sub_start);
            } else {
                clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
                bt_fpga_gemv_dequant(s->q, lw->wq.tmac, &bt_fpga_layer_locs[l].wq, inv_scale);
                bt_fpga_compare_tmac("wq", l, lw->wq.tmac, s->q8_buf, inv_scale, s->q);
                bt_profile_accum_elapsed(
                    &s->profile_last_qkv_stage_sec[BT_PROFILE_QKV_STAGE_WQ],
                    ts_sub_start);
                clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
                bt_fpga_gemv_dequant(s->k, lw->wk.tmac, &bt_fpga_layer_locs[l].wk, inv_scale);
                bt_fpga_compare_tmac("wk", l, lw->wk.tmac, s->q8_buf, inv_scale, s->k);
                bt_profile_accum_elapsed(
                    &s->profile_last_qkv_stage_sec[BT_PROFILE_QKV_STAGE_WK],
                    ts_sub_start);
                clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
                bt_fpga_gemv_dequant(s->v, lw->wv.tmac, &bt_fpga_layer_locs[l].wv, inv_scale);
                bt_fpga_compare_tmac("wv", l, lw->wv.tmac, s->q8_buf, inv_scale, s->v);
                bt_profile_accum_elapsed(
                    &s->profile_last_qkv_stage_sec[BT_PROFILE_QKV_STAGE_WV],
                    ts_sub_start);
            }
        } else
#endif
        {
            bt_tmac_weight_t* tw = lw->wq.tmac;
            clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
            tmac_build_three_lut(s->lut_buf, s->q8_buf, tw->n3);
            int16_t* two_lut = s->lut_buf + tw->n3 * 16;
            tmac_build_two_lut(two_lut, s->q8_buf, tw->n3 * 3, tw->cols, tw->n2);
            bt_profile_accum_elapsed(
                &s->profile_last_qkv_stage_sec[BT_PROFILE_QKV_STAGE_PREP],
                ts_sub_start);

            clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
            tmac_gemv(s->q, tw, inv_scale, s->lut_buf, two_lut);
            bt_profile_accum_elapsed(
                &s->profile_last_qkv_stage_sec[BT_PROFILE_QKV_STAGE_WQ],
                ts_sub_start);
            clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
            tmac_gemv(s->k, lw->wk.tmac, inv_scale, s->lut_buf, two_lut);
            bt_profile_accum_elapsed(
                &s->profile_last_qkv_stage_sec[BT_PROFILE_QKV_STAGE_WK],
                ts_sub_start);
            clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
            tmac_gemv(s->v, lw->wv.tmac, inv_scale, s->lut_buf, two_lut);
            bt_profile_accum_elapsed(
                &s->profile_last_qkv_stage_sec[BT_PROFILE_QKV_STAGE_WV],
                ts_sub_start);
        }
        bt_profile_accum_elapsed(
            &s->profile_last_layer_stage_sec[BT_PROFILE_LAYER_STAGE_QKV],
            ts_stage_start);

        /* Stage 8: RoPE on Q and K */
        clock_gettime(CLOCK_MONOTONIC, &ts_stage_start);
        rope(s->q, head_dim, n_heads, pos, cfg->rope_theta);
        rope(s->k, head_dim, n_kv_heads, pos, cfg->rope_theta);
        bt_profile_accum_elapsed(
            &s->profile_last_layer_stage_sec[BT_PROFILE_LAYER_STAGE_ROPE],
            ts_stage_start);

        /* Stage 9: Store K/V into TurboQuant cache */
        clock_gettime(CLOCK_MONOTONIC, &ts_stage_start);
        for (int h = 0; h < n_kv_heads; h++) {
            size_t idx = (size_t)h * cfg->max_seq_len + (size_t)pos;
            tq_quantize(&kv->k_cache[idx], s->k + h * head_dim, head_dim);
            tq_quantize(&kv->v_cache[idx], s->v + h * head_dim, head_dim);
        }
        bt_profile_accum_elapsed(
            &s->profile_last_layer_stage_sec[BT_PROFILE_LAYER_STAGE_KV_CACHE],
            ts_stage_start);

        /* Stage 10: Attention scores (GQA with TurboQuant) */
        clock_gettime(CLOCK_MONOTONIC, &ts_stage_start);
        int seq_len = pos + 1;
        float att_scale = 1.0f / sqrtf((float)head_dim);
        for (int qh = 0; qh < n_heads; qh++) {
            float* qhead = s->q + qh * head_dim;
            int kv_head = qh / gqa_groups;
            float* att = s->att + (size_t)qh * cfg->max_seq_len;

            /* Precompute RHT(query) and QJL projection ONCE per query head */
            memcpy(s->q_rht, qhead, head_dim * sizeof(float));
            tq_rht_forward(s->q_rht, head_dim, TQ_DEFAULT_SEED);
            for (int j = 0; j < head_dim; j++) {
                float proj = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    proj += s->q_rht[d] * tq_qjl_entry(d, j);
                s->q_qjl[j] = proj;
            }

            /* Two-stage dot product for each cached key position */
            for (int t = 0; t < seq_len; t++) {
                size_t ki = (size_t)kv_head * cfg->max_seq_len + (size_t)t;
                att[t] = tq_attention_dot(s->q_rht, s->q_qjl,
                                          &kv->k_cache[ki], head_dim)
                       * att_scale;
            }
            softmax(att, seq_len);

            /* Weighted V sum with MSE-only dequant */
            float* out_h = s->xb2 + qh * head_dim;
            memset(out_h, 0, head_dim * sizeof(float));
            for (int t = 0; t < seq_len; t++) {
                if (att[t] < 1e-8f) continue;
                size_t vi = (size_t)kv_head * cfg->max_seq_len + (size_t)t;
                tq_dequant_accum(out_h, att[t], &kv->v_cache[vi], head_dim);
            }
        }
        bt_profile_accum_elapsed(
            &s->profile_last_layer_stage_sec[BT_PROFILE_LAYER_STAGE_ATTENTION],
            ts_stage_start);

        /* Stage 11: Attention sub-norm + output projection */
        clock_gettime(CLOCK_MONOTONIC, &ts_stage_start);
        rms_norm(s->xb, s->xb2, lw->attn_sub_norm, dim, cfg->norm_eps);
#ifdef BT_FPGA
        if (bt_fpga_regs) {
            inv_scale = bitlinear_quantize(s->q8_buf, s->xb, dim);
            bt_fpga_upload_activations(s->q8_buf, dim);
            if (model->btpk) {
                bt_fpga_gemv_dequant_btpk(s->xb2, &model->btpk->layers[l].wo,
                                          &bt_fpga_layer_locs[l].wo, inv_scale);
            } else {
                bt_fpga_gemv_dequant(s->xb2, lw->wo.tmac,
                                     &bt_fpga_layer_locs[l].wo, inv_scale);
                bt_fpga_compare_tmac("wo", l, lw->wo.tmac, s->q8_buf, inv_scale, s->xb2);
            }
        } else
#endif
        tmac_forward(s->xb2, s->xb, s->q8_buf, s->lut_buf, &lw->wo);
        bt_profile_accum_elapsed(
            &s->profile_last_layer_stage_sec[BT_PROFILE_LAYER_STAGE_ATTN_OUT],
            ts_stage_start);

        /* Residual */
        clock_gettime(CLOCK_MONOTONIC, &ts_stage_start);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
        bt_profile_accum_elapsed(
            &s->profile_last_layer_stage_sec[BT_PROFILE_LAYER_STAGE_ATTN_RESIDUAL],
            ts_stage_start);

        /* Stage 12: FFN pre-norm + gate/up projections */
        clock_gettime(CLOCK_MONOTONIC, &ts_stage_start);
        rms_norm(s->xb, s->x, lw->ffn_norm, dim, cfg->norm_eps);
        inv_scale = bitlinear_quantize(s->q8_buf, s->xb, dim);
#ifdef BT_FPGA
        if (bt_fpga_regs) {
            bt_fpga_upload_activations(s->q8_buf, dim);
            if (model->btpk) {
                btpk_layer_t* bl = &model->btpk->layers[l];
                bt_fpga_gemv_dequant_btpk(s->hb,  &bl->w_gate,
                                          &bt_fpga_layer_locs[l].w_gate, inv_scale);
                bt_fpga_gemv_dequant_btpk(s->hb2, &bl->w_up,
                                          &bt_fpga_layer_locs[l].w_up, inv_scale);
            } else {
                bt_fpga_gemv_dequant(s->hb,  lw->w_gate.tmac, &bt_fpga_layer_locs[l].w_gate, inv_scale);
                bt_fpga_compare_tmac("w_gate", l, lw->w_gate.tmac, s->q8_buf, inv_scale, s->hb);
                bt_fpga_gemv_dequant(s->hb2, lw->w_up.tmac,   &bt_fpga_layer_locs[l].w_up,   inv_scale);
                bt_fpga_compare_tmac("w_up", l, lw->w_up.tmac, s->q8_buf, inv_scale, s->hb2);
            }
        } else
#endif
        {
            bt_tmac_weight_t* tw = lw->w_gate.tmac;
            tmac_build_three_lut(s->lut_buf, s->q8_buf, tw->n3);
            int16_t* two_lut = s->lut_buf + tw->n3 * 16;
            tmac_build_two_lut(two_lut, s->q8_buf, tw->n3 * 3, tw->cols, tw->n2);
            tmac_gemv(s->hb, tw, inv_scale, s->lut_buf, two_lut);
            tmac_gemv(s->hb2, lw->w_up.tmac, inv_scale, s->lut_buf, two_lut);
        }
        bt_profile_accum_elapsed(
            &s->profile_last_layer_stage_sec[BT_PROFILE_LAYER_STAGE_FFN_IN],
            ts_stage_start);

        /* SqReLU gating: hidden = SqReLU(gate) * up */
        clock_gettime(CLOCK_MONOTONIC, &ts_stage_start);
#ifdef __ARM_NEON
        {
            float32x4_t zero = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i + 3 < ffn_dim; i += 4) {
                float32x4_t g = vld1q_f32(s->hb + i);
                float32x4_t u = vld1q_f32(s->hb2 + i);
                float32x4_t r = vmaxq_f32(g, zero);
                vst1q_f32(s->hb + i, vmulq_f32(vmulq_f32(r, r), u));
            }
            for (; i < ffn_dim; i++) {
                float g = s->hb[i];
                float r = (g > 0.0f) ? g : 0.0f;
                s->hb[i] = r * r * s->hb2[i];
            }
        }
#else
        for (int i = 0; i < ffn_dim; i++) {
            float g = s->hb[i];
            float r = (g > 0.0f) ? g : 0.0f;
            s->hb[i] = r * r * s->hb2[i];
        }
#endif
        bt_profile_accum_elapsed(
            &s->profile_last_layer_stage_sec[BT_PROFILE_LAYER_STAGE_SQRELU],
            ts_stage_start);

        /* Stage 13: FFN sub-norm + down projection */
        clock_gettime(CLOCK_MONOTONIC, &ts_stage_start);
        rms_norm(s->hb2, s->hb, lw->ffn_sub_norm, ffn_dim, cfg->norm_eps);
#ifdef BT_FPGA
        if (bt_fpga_regs) {
            inv_scale = bitlinear_quantize(s->q8_buf, s->hb2, ffn_dim);
            bt_fpga_upload_activations(s->q8_buf, ffn_dim);
            if (model->btpk) {
                bt_fpga_gemv_dequant_btpk(s->xb, &model->btpk->layers[l].w_down,
                                          &bt_fpga_layer_locs[l].w_down, inv_scale);
            } else {
                bt_fpga_gemv_dequant(s->xb, lw->w_down.tmac,
                                     &bt_fpga_layer_locs[l].w_down, inv_scale);
                bt_fpga_compare_tmac("w_down", l, lw->w_down.tmac, s->q8_buf, inv_scale, s->xb);
            }
        } else
#endif
        tmac_forward(s->xb, s->hb2, s->q8_buf, s->lut_buf, &lw->w_down);
        bt_profile_accum_elapsed(
            &s->profile_last_layer_stage_sec[BT_PROFILE_LAYER_STAGE_FFN_OUT],
            ts_stage_start);

        /* Residual */
        clock_gettime(CLOCK_MONOTONIC, &ts_stage_start);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
        bt_profile_accum_elapsed(
            &s->profile_last_layer_stage_sec[BT_PROFILE_LAYER_STAGE_FFN_RESIDUAL],
            ts_stage_start);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_layers_end);
    s->profile_last_layers_sec =
        bt_timespec_elapsed_sec(ts_layers_start, ts_layers_end);

    /* Stage 14: Final norm + LM head (tied to token embedding) */
    clock_gettime(CLOCK_MONOTONIC, &ts_lm_head_start);
    rms_norm(s->xb, s->x, w->final_norm, dim, cfg->norm_eps);

    /* LM head: logits[v] = dot(xb, token_embd[v]) for all vocab
     * token_embd is Q6_K [vocab_size][dim] */
    {
        int nb_per_row = dim / BT_QK_K;
        size_t row_bytes = (size_t)nb_per_row * sizeof(bt_block_q6k_t);
        float* tmp = s->xb2;  /* reuse scratch buffer (>= dim floats) */
        for (int v = 0; v < cfg->vocab_size; v++) {
            const bt_block_q6k_t* ev = (const bt_block_q6k_t*)(w->token_embedding + (size_t)v * row_bytes);
            dequantize_q6k(ev, tmp, dim);
            float sum = 0.0f;
#ifdef __ARM_NEON
            float32x4_t acc = vdupq_n_f32(0.0f);
            int d = 0;
            for (; d + 3 < dim; d += 4) {
                float32x4_t xv = vld1q_f32(s->xb + d);
                float32x4_t tv = vld1q_f32(tmp + d);
#ifdef __aarch64__
                acc = vfmaq_f32(acc, xv, tv);
#else
                acc = vmlaq_f32(acc, xv, tv);
#endif
            }
#ifdef __aarch64__
            sum = vaddvq_f32(acc);
#else
            {
                float32x2_t s2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
                s2 = vpadd_f32(s2, s2);
                sum = vget_lane_f32(s2, 0);
            }
#endif
            for (; d < dim; d++) sum += s->xb[d] * tmp[d];
#else
            for (int d = 0; d < dim; d++) sum += s->xb[d] * tmp[d];
#endif
            s->logits[v] = sum;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_lm_head_end);
    s->profile_last_lm_head_sec =
        bt_timespec_elapsed_sec(ts_lm_head_start, ts_lm_head_end);
}

/* ================================================================
 * §9. SAMPLING (Stage 15)
 * ================================================================ */

void bt_sampler_init(bt_sampler_t* s, float temperature, float top_p,
                     uint64_t seed) {
    s->temperature = temperature;
    s->top_p = top_p;
    s->rng_state = seed ? seed : (uint64_t)time(NULL);
}

static float bt_random(bt_sampler_t* s) {
    uint64_t x = s->rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    s->rng_state = x;
    return (float)((x * 0x2545F4914F6CDD1DULL) >> 40) / (float)(1 << 24);
}

typedef struct { float prob; int idx; } bt_pi_t;
static int pi_cmp(const void* a, const void* b) {
    float pa = ((const bt_pi_t*)a)->prob;
    float pb = ((const bt_pi_t*)b)->prob;
    return (pa < pb) - (pa > pb);
}

int bt_sample(bt_sampler_t* s, const float* logits, int vocab_size) {
    if (s->temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    float* probs = (float*)bt_calloc(vocab_size, sizeof(float));
    for (int i = 0; i < vocab_size; i++)
        probs[i] = logits[i] / s->temperature;
    softmax(probs, vocab_size);

    int result = 0;
    if (s->top_p < 1.0f) {
        bt_pi_t* sorted = (bt_pi_t*)bt_calloc(vocab_size, sizeof(bt_pi_t));
        for (int i = 0; i < vocab_size; i++) {
            sorted[i].prob = probs[i];
            sorted[i].idx = i;
        }
        qsort(sorted, vocab_size, sizeof(bt_pi_t), pi_cmp);

        float cumsum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += sorted[i].prob;
            if (cumsum >= s->top_p) { cutoff = i + 1; break; }
        }
        float r = bt_random(s) * cumsum;
        float cdf = 0.0f;
        result = sorted[0].idx;
        for (int i = 0; i < cutoff; i++) {
            cdf += sorted[i].prob;
            if (r <= cdf) { result = sorted[i].idx; break; }
        }
        free(sorted);
    } else {
        float r = bt_random(s);
        float cdf = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cdf += probs[i];
            if (r <= cdf) { result = i; break; }
        }
    }
    free(probs);
    return result;
}

/* ================================================================
 * §10. TOKENIZER (Stage 1) — BPE from GGUF metadata
 *
 * GPT-2 BPE uses a byte-to-unicode mapping so all 256 byte values
 * can be represented as unique Unicode characters in the vocabulary:
 *   - Printable ASCII 0x21-0x7E → identity (same codepoint)
 *   - Latin-1 0xA1-0xAC, 0xAE-0xFF → identity
 *   - Everything else (0x00-0x20, 0x7F-0xA0, 0xAD) → U+0100..U+0143
 * Notably: space (0x20) → U+0120 (Ġ), newline (0x0A) → U+010A (Ċ)
 *
 * Encoding: raw bytes → byte-to-unicode → per-byte vocab lookup → BPE merge
 * Decoding: token string → unicode-to-byte reverse mapping → raw bytes
 * ================================================================ */

/* Convert a raw byte to its GPT-2 BPE UTF-8 string (1-3 bytes + NUL).
 * Returns the number of bytes written (excluding NUL). */
static int byte_to_bpe(unsigned char byte, char out[4]) {
    if (byte >= 0x21 && byte <= 0x7E) {
        /* Printable ASCII: maps to itself */
        out[0] = (char)byte;
        out[1] = '\0';
        return 1;
    }
    /* Compute the Unicode codepoint for this byte */
    int cp;
    if ((byte >= 0xA1 && byte <= 0xAC) || (byte >= 0xAE)) {
        /* Latin-1 printable: identity mapping */
        cp = byte;
    } else if (byte <= 0x20) {
        /* 0x00-0x20 (includes space) → U+0100..U+0120 */
        cp = 0x100 + byte;
    } else if (byte == 0x7F) {
        cp = 0x121;  /* DEL → U+0121 */
    } else if (byte >= 0x80 && byte <= 0xA0) {
        cp = 0x122 + (byte - 0x80);  /* 0x80-0xA0 → U+0122..U+0142 */
    } else {
        cp = 0x143;  /* 0xAD (soft hyphen) → U+0143 */
    }
    /* Encode codepoint as UTF-8 (all are <= U+0143, so 2 bytes) */
    out[0] = (char)(0xC0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3F));
    out[2] = '\0';
    return 2;
}

/* Reverse mapping: BPE codepoints U+0100..U+0143 → raw byte values */
static const uint8_t BPE_N2B[68] = {
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07, /* U+0100-0107 */
    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F, /* U+0108-010F */
    0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17, /* U+0110-0117 */
    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F, /* U+0118-011F */
    0x20,                                      /* U+0120 = space */
    0x7F,                                      /* U+0121 = DEL */
    0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87, /* U+0122-0129 */
    0x88,0x89,0x8A,0x8B,0x8C,0x8D,0x8E,0x8F, /* U+012A-0131 */
    0x90,0x91,0x92,0x93,0x94,0x95,0x96,0x97, /* U+0132-0139 */
    0x98,0x99,0x9A,0x9B,0x9C,0x9D,0x9E,0x9F, /* U+013A-0141 */
    0xA0,                                      /* U+0142 = NBSP */
    0xAD,                                      /* U+0143 = soft hyphen */
};

/* Binary search vocab lookup using sorted index. O(log V). */
static int tok_lookup(const bt_tokenizer_t* tok, const char* str) {
    if (!tok->sorted_idx) return -1;
    int lo = 0, hi = tok->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = strcmp(str, tok->vocab[tok->sorted_idx[mid]]);
        if (cmp == 0) return tok->sorted_idx[mid];
        if (cmp < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return -1;
}

int bt_encode(const bt_tokenizer_t* tok, const char* text,
              int* tokens, int max_tokens, int add_bos) {
    if (!text || !tokens) return 0;

    int text_len = (int)strlen(text);
    if (text_len == 0 && !add_bos) return 0;

    /* Work buffer for per-byte tokens before BPE merging */
    int* work = (int*)malloc((text_len + 1) * sizeof(int));
    if (!work) return 0;
    int n_work = 0;

    /* Step 1: Convert each raw byte to its BPE unicode token.
     * "Where is" → bytes [57 68 65 72 65 20 69 73]
     *            → BPE   [W  h  e  r  e  Ġ  i  s ]
     *            → tokens [54 71 68 81 68 220 72 82] */
    for (int i = 0; i < text_len; i++) {
        char bpe[4];
        byte_to_bpe((unsigned char)text[i], bpe);
        int id = tok_lookup(tok, bpe);
        if (id >= 0) {
            work[n_work++] = id;
        }
        /* If not found, skip byte (shouldn't happen with valid vocab) */
    }

    /* Step 2: BPE merge loop — greedily merge the highest-scored adjacent pair.
     * [W, h, e, r, e, Ġ, i, s] → [Wh, e, r, e, Ġ, is]
     * → [Wh, ere, Ġ, is] → [Where, Ġis] → done (no more merges) */
    char merge_buf[1024];
    while (n_work >= 2) {
        float best_score = -1e30f;
        int best_idx = -1, best_tok = -1;

        for (int i = 0; i < n_work - 1; i++) {
            const char* s1 = tok->vocab[work[i]];
            const char* s2 = tok->vocab[work[i + 1]];
            int l1 = (int)strlen(s1);
            int l2 = (int)strlen(s2);
            if (l1 + l2 >= (int)sizeof(merge_buf)) continue;
            memcpy(merge_buf, s1, l1);
            memcpy(merge_buf + l1, s2, l2);
            merge_buf[l1 + l2] = '\0';

            int id = tok_lookup(tok, merge_buf);
            if (id >= 0 && tok->scores[id] > best_score) {
                best_score = tok->scores[id];
                best_idx = i;
                best_tok = id;
            }
        }
        if (best_idx < 0) break;

        work[best_idx] = best_tok;
        for (int i = best_idx + 1; i < n_work - 1; i++)
            work[i] = work[i + 1];
        n_work--;
    }

    /* Step 3: Copy to output with optional BOS prefix */
    int n = 0;
    if (add_bos && n < max_tokens)
        tokens[n++] = tok->bos_id;
    for (int i = 0; i < n_work && n < max_tokens; i++)
        tokens[n++] = work[i];

    free(work);
    return n;
}

/* Decode token ID to raw text.
 * Reverses the GPT-2 byte-to-unicode mapping:
 *   Ġ (U+0120) → space, Ċ (U+010A) → newline, etc.
 * All codepoints in U+0100..U+0143 are looked up in BPE_N2B. */
const char* bt_decode(const bt_tokenizer_t* tok, int prev_token, int token) {
    static char buf[1024];
    if (token < 0 || token >= tok->vocab_size) return "";
    const unsigned char* src = (const unsigned char*)tok->vocab[token];

    int j = 0;
    while (*src && j < (int)sizeof(buf) - 4) {
        if (src[0] < 0x80) {
            /* ASCII: identity mapping */
            buf[j++] = (char)src[0];
            src++;
        } else if ((src[0] & 0xE0) == 0xC0 && (src[1] & 0xC0) == 0x80) {
            /* 2-byte UTF-8: decode codepoint */
            int cp = ((src[0] & 0x1F) << 6) | (src[1] & 0x3F);
            if (cp >= 0x100 && cp <= 0x143) {
                /* BPE remapped byte: look up raw value */
                buf[j++] = (char)BPE_N2B[cp - 0x100];
            } else if ((cp >= 0xA1 && cp <= 0xAC) || (cp >= 0xAE && cp <= 0xFF)) {
                /* Latin-1 identity: output as raw byte */
                buf[j++] = (char)cp;
            } else {
                /* Other 2-byte codepoint: pass through as UTF-8 */
                buf[j++] = (char)src[0];
                buf[j++] = (char)src[1];
            }
            src += 2;
        } else {
            /* 3+ byte UTF-8: pass through */
            buf[j++] = (char)*src++;
        }
    }
    buf[j] = '\0';

    const char* out = buf;
    if (prev_token == tok->bos_id && out[0] == ' ') out++;
    return out;
}

/* ================================================================
 * §11. GGUF MODEL LOADING
 * ================================================================ */

static void init_i2s_weight(bt_i2s_weight_t* w, const gguf_tensor_info_t* ti,
                            const uint8_t* data_base) {
    w->cols = (int)ti->dims[0];  /* ne0 = input features */
    w->rows = (int)ti->dims[1];  /* ne1 = output features */

    const uint8_t* blob = data_base + ti->offset;
    size_t n_elem = (size_t)w->rows * w->cols;
    size_t packed_bytes = n_elem / 4;  /* 4 values per byte */

    w->data = blob;
    /* Scale is the last 4 bytes of the I2_S blob */
    memcpy(&w->scale, blob + packed_bytes, sizeof(float));
    w->tmac = NULL;
}

static float* load_f32_tensor(const gguf_tensor_info_t* ti,
                               const uint8_t* data_base) {
    size_t n = 1;
    for (uint32_t d = 0; d < ti->n_dims; d++) n *= ti->dims[d];
    float* out = (float*)bt_calloc(n, sizeof(float));
    memcpy(out, data_base + ti->offset, n * sizeof(float));
    return out;
}

static int alloc_state(bt_state_t* s, const bt_config_t* cfg) {
    int dim = cfg->dim;
    int kv_dim = BT_KV_DIM(cfg);
    int ffn_dim = cfg->ffn_dim;
    int max_buf = dim;
    if (kv_dim > max_buf) max_buf = kv_dim;
    if (ffn_dim > max_buf) max_buf = ffn_dim;

    s->x      = (float*)bt_calloc(dim, sizeof(float));
    s->xb     = (float*)bt_calloc(dim, sizeof(float));
    s->xb2    = (float*)bt_calloc(dim, sizeof(float));
    s->hb     = (float*)bt_calloc(ffn_dim, sizeof(float));
    s->hb2    = (float*)bt_calloc(ffn_dim, sizeof(float));
    s->q      = (float*)bt_calloc(dim, sizeof(float));
    s->k      = (float*)bt_calloc(kv_dim, sizeof(float));
    s->v      = (float*)bt_calloc(kv_dim, sizeof(float));
    s->att    = (float*)bt_calloc((size_t)cfg->n_heads * cfg->max_seq_len,
                                  sizeof(float));
    s->q8_buf = (int8_t*)bt_calloc(max_buf, sizeof(int8_t));
    /* T-MAC LUT: max_k/3 three-groups * 16 entries + 16 for two-group */
    int max_k = dim > ffn_dim ? dim : ffn_dim;
    s->lut_buf = (int16_t*)bt_calloc((size_t)(max_k / 3 + 1) * 16, sizeof(int16_t));
    int head_dim = BT_HEAD_DIM(cfg);
    s->q_rht  = (float*)bt_calloc(head_dim, sizeof(float));
    s->q_qjl  = (float*)bt_calloc(head_dim, sizeof(float));
    s->logits = (float*)bt_calloc(cfg->vocab_size, sizeof(float));

    /* TurboQuant KV cache: 1 block per head (head_dim=128=BT_QK) */
    int bph = (head_dim + BT_QK - 1) / BT_QK;

    s->kv = (bt_kv_cache_t*)bt_calloc(cfg->n_layers, sizeof(bt_kv_cache_t));
    for (int l = 0; l < cfg->n_layers; l++) {
        s->kv[l].blocks_per_head = bph;
        size_t total = (size_t)cfg->n_kv_heads * cfg->max_seq_len * bph;
        s->kv[l].k_cache = (bt_kv_block_t*)bt_calloc(total, sizeof(bt_kv_block_t));
        s->kv[l].v_cache = (bt_kv_block_t*)bt_calloc(total, sizeof(bt_kv_block_t));
    }
    return 0;
}

/* ================================================================
 * §11b. .btpk LOADER — pre-packed, FPGA-native layout
 *
 * Only reachable on BT_FPGA builds: CPU-path T-MAC GEMV reads the
 * 2-nib/8-sign CPU layout (bt_tmac_weight_t), which is not present
 * in the .btpk format.
 * ================================================================ */

#ifdef BT_FPGA
static int btpk_check_range(const bt_model_t* model,
                            uint64_t off, uint64_t size,
                            const char* what) {
    uint64_t file_size = (uint64_t)model->mmap_size;
    if (off > file_size || size > file_size - off) {
        fprintf(stderr,
                "biturbo: .btpk %s range out of bounds "
                "(off=%llu size=%llu file=%llu)\n",
                what,
                (unsigned long long)off,
                (unsigned long long)size,
                (unsigned long long)file_size);
        return -1;
    }
    return 0;
}

static int btpk_check_weight_dir(const bt_model_t* model,
                                 const btpk_weight_dir_t* wd,
                                 const char* what) {
    uint64_t expect_nib = (uint64_t)wd->rows * (uint64_t)wd->nib_stride;
    uint64_t expect_sign = (uint64_t)wd->rows * (uint64_t)wd->sign_stride;

    if (wd->rows == 0 || wd->cols == 0 || wd->nib_stride <= 0 || wd->sign_stride <= 0) {
        fprintf(stderr, "biturbo: .btpk %s has invalid dimensions/strides\n", what);
        return -1;
    }
    if (wd->k_padded <= 0 || (wd->k_padded % 3) != 0 || wd->n3 != (wd->k_padded / 3)) {
        fprintf(stderr, "biturbo: .btpk %s has inconsistent tail packing metadata\n", what);
        return -1;
    }
    if (wd->nib_size != expect_nib || wd->sign_size != expect_sign) {
        fprintf(stderr,
                "biturbo: .btpk %s size/stride mismatch "
                "(nib=%llu expect=%llu sign=%llu expect=%llu)\n",
                what,
                (unsigned long long)wd->nib_size,
                (unsigned long long)expect_nib,
                (unsigned long long)wd->sign_size,
                (unsigned long long)expect_sign);
        return -1;
    }
    if (btpk_check_range(model, wd->nib_off, wd->nib_size, what) != 0) {
        return -1;
    }
    if (btpk_check_range(model, wd->sign_off, wd->sign_size, what) != 0) {
        return -1;
    }
    return 0;
}

static int bt_load_btpk(bt_model_t* model) {
    if (model->mmap_size < sizeof(btpk_header_t)) {
        fprintf(stderr, "biturbo: .btpk file truncated\n");
        return -1;
    }
    const btpk_header_t* h = (const btpk_header_t*)model->mmap_data;

    if (h->version != BTPK_VERSION) {
        fprintf(stderr,
                "biturbo: .btpk version %u, expected %u "
                "(repack the model with the current pack_btpk; v1 dropped K%%3 tails)\n",
                h->version, BTPK_VERSION);
        return -1;
    }
    if (h->format != BTPK_FMT_TMAC_TL2) {
        fprintf(stderr, "biturbo: .btpk format %u unsupported\n", h->format);
        return -1;
    }
    if (h->num_engines != 32 || h->beat_bytes != 16) {
        fprintf(stderr, "biturbo: .btpk striped for engines=%u beat=%u "
                "but this build expects 32/16\n",
                h->num_engines, h->beat_bytes);
        return -1;
    }
    if (h->total_file_size != model->mmap_size) {
        fprintf(stderr, "biturbo: .btpk size mismatch (%llu vs on-disk %zu)\n",
                (unsigned long long)h->total_file_size, model->mmap_size);
        return -1;
    }

    bt_config_t* cfg = &model->config;
    cfg->dim         = h->dim;
    cfg->n_layers    = h->n_layers;
    cfg->n_heads     = h->n_heads;
    cfg->n_kv_heads  = h->n_kv_heads;
    cfg->vocab_size  = h->vocab_size;
    cfg->ffn_dim     = h->ffn_dim;
    cfg->max_seq_len = h->max_seq_len;
    cfg->norm_eps    = h->norm_eps;
    cfg->rope_theta  = h->rope_theta;

    if (btpk_check_range(model, h->tokenizer_off, h->tokenizer_size, "tokenizer") != 0)
        return -1;
    if (btpk_check_range(model, h->layers_off,
                         (uint64_t)cfg->n_layers * sizeof(btpk_layer_dir_t),
                         "layer directory") != 0)
        return -1;

    {
        uint64_t expected_embed_size =
            (uint64_t)cfg->vocab_size *
            (uint64_t)((cfg->dim + BT_QK_K - 1) / BT_QK_K) *
            (uint64_t)sizeof(bt_block_q6k_t);
        uint64_t expected_final_norm_size =
            (uint64_t)cfg->dim * sizeof(float);

        if (h->token_embed_size < expected_embed_size) {
            fprintf(stderr,
                    "biturbo: .btpk token embedding too small "
                    "(%llu < %llu)\n",
                    (unsigned long long)h->token_embed_size,
                    (unsigned long long)expected_embed_size);
            return -1;
        }
        if (h->final_norm_size < expected_final_norm_size) {
            fprintf(stderr,
                    "biturbo: .btpk final_norm too small "
                    "(%llu < %llu)\n",
                    (unsigned long long)h->final_norm_size,
                    (unsigned long long)expected_final_norm_size);
            return -1;
        }
        if (btpk_check_range(model, h->token_embed_off, h->token_embed_size,
                             "token embedding") != 0)
            return -1;
        if (btpk_check_range(model, h->final_norm_off, h->final_norm_size,
                             "final_norm") != 0)
            return -1;
    }

    fprintf(stderr, "biturbo: .btpk v%u — dim=%d layers=%d heads=%d "
            "kv_heads=%d vocab=%d ffn=%d ctx=%d\n",
            h->version, cfg->dim, cfg->n_layers, cfg->n_heads,
            cfg->n_kv_heads, cfg->vocab_size, cfg->ffn_dim, cfg->max_seq_len);

    /* Tokenizer: parse from blob [scores][(len,bytes)*] */
    bt_tokenizer_t* tok = &model->tokenizer;
    tok->vocab_size    = h->tok_vocab_size;
    tok->max_token_len = h->tok_max_token_len;
    tok->bos_id        = h->tok_bos_id;
    tok->eos_id        = h->tok_eos_id;
    tok->eot_id        = h->tok_eot_id;
    tok->vocab  = (char**)bt_calloc(tok->vocab_size, sizeof(char*));
    tok->scores = (float*)bt_calloc(tok->vocab_size, sizeof(float));

    const uint8_t* tb  = model->mmap_data + h->tokenizer_off;
    const uint8_t* tbe = tb + h->tokenizer_size;
    if (h->tokenizer_size < (uint64_t)tok->vocab_size * sizeof(float)) {
        fprintf(stderr, "biturbo: .btpk tokenizer blob too small for scores\n");
        return -1;
    }
    memcpy(tok->scores, tb, (size_t)tok->vocab_size * sizeof(float));
    const uint8_t* p = tb + (size_t)tok->vocab_size * sizeof(float);
    for (int i = 0; i < tok->vocab_size; i++) {
        if (p + 2 > tbe) {
            fprintf(stderr, "biturbo: .btpk tokenizer truncated at %d\n", i);
            return -1;
        }
        uint16_t len;
        memcpy(&len, p, 2); p += 2;
        if (p + len > tbe) {
            fprintf(stderr, "biturbo: .btpk tokenizer bad len at %d\n", i);
            return -1;
        }
        tok->vocab[i] = (char*)bt_calloc(len + 1, 1);
        memcpy(tok->vocab[i], p, len);
        tok->vocab[i][len] = '\0';
        p += len;
    }
    /* Build sorted vocab index */
    tok->sorted_idx = (int*)bt_calloc(tok->vocab_size, sizeof(int));
    for (int i = 0; i < tok->vocab_size; i++) tok->sorted_idx[i] = i;
    bt_sort_vocab_global = tok->vocab;
    qsort(tok->sorted_idx, tok->vocab_size, sizeof(int), bt_sort_vocab_cmp);

    fprintf(stderr, "biturbo: tokenizer: %d tokens, bos=%d eos=%d eot=%d\n",
            tok->vocab_size, tok->bos_id, tok->eos_id, tok->eot_id);

    /* Allocate state once config is populated */
    alloc_state(&model->state, cfg);

    /* Point weights at mmap'd blobs (zero copy, zero transform) */
    bt_weights_t* wt = &model->weights;
    wt->token_embedding = model->mmap_data + h->token_embed_off;

    wt->final_norm = (float*)bt_calloc(cfg->dim, sizeof(float));
    memcpy(wt->final_norm,
           model->mmap_data + h->final_norm_off,
           (size_t)cfg->dim * sizeof(float));

    /* Per-layer norms: copied to heap (they're small F32 arrays and
     * existing code treats them as owned buffers freed in bt_free_model). */
    wt->layers = (bt_layer_weights_t*)bt_calloc(cfg->n_layers,
                                                 sizeof(bt_layer_weights_t));

    const btpk_layer_dir_t* lds =
        (const btpk_layer_dir_t*)(model->mmap_data + h->layers_off);

    model->btpk = (bt_btpk_weights_t*)bt_calloc(1, sizeof(*model->btpk));
    model->btpk->layers = (btpk_layer_t*)bt_calloc(cfg->n_layers,
                                                   sizeof(btpk_layer_t));

    #define LOAD_F32(dst_field, off_field, count) do { \
        lw->dst_field = (float*)bt_calloc((count), sizeof(float)); \
        memcpy(lw->dst_field, model->mmap_data + lds[l].off_field, \
               (size_t)(count) * sizeof(float)); \
    } while (0)

    #define LOAD_BTPK_W(out, src) do { \
        (out)->rows        = (int)lds[l].src.rows; \
        (out)->cols        = (int)lds[l].src.cols; \
        (out)->n3          = lds[l].src.n3; \
        (out)->k_padded    = lds[l].src.k_padded; \
        (out)->nib_stride  = lds[l].src.nib_stride; \
        (out)->sign_stride = lds[l].src.sign_stride; \
        (out)->nib_size    = lds[l].src.nib_size; \
        (out)->sign_size   = lds[l].src.sign_size; \
        (out)->nib_data    = model->mmap_data + lds[l].src.nib_off; \
        (out)->sign_data   = model->mmap_data + lds[l].src.sign_off; \
        (out)->scale       = lds[l].src.scale; \
    } while (0)

    for (int l = 0; l < cfg->n_layers; l++) {
        bt_layer_weights_t* lw = &wt->layers[l];
        btpk_layer_t*       bl = &model->btpk->layers[l];
        char what[64];

        snprintf(what, sizeof(what), "layer %d attn_norm", l);
        if (btpk_check_range(model, lds[l].attn_norm_off,
                             (uint64_t)cfg->dim * sizeof(float), what) != 0)
            return -1;
        snprintf(what, sizeof(what), "layer %d attn_sub_norm", l);
        if (btpk_check_range(model, lds[l].attn_sub_norm_off,
                             (uint64_t)cfg->dim * sizeof(float), what) != 0)
            return -1;
        snprintf(what, sizeof(what), "layer %d ffn_norm", l);
        if (btpk_check_range(model, lds[l].ffn_norm_off,
                             (uint64_t)cfg->dim * sizeof(float), what) != 0)
            return -1;
        snprintf(what, sizeof(what), "layer %d ffn_sub_norm", l);
        if (btpk_check_range(model, lds[l].ffn_sub_norm_off,
                             (uint64_t)cfg->ffn_dim * sizeof(float), what) != 0)
            return -1;

        snprintf(what, sizeof(what), "layer %d wq", l);
        if (btpk_check_weight_dir(model, &lds[l].wq, what) != 0) return -1;
        snprintf(what, sizeof(what), "layer %d wk", l);
        if (btpk_check_weight_dir(model, &lds[l].wk, what) != 0) return -1;
        snprintf(what, sizeof(what), "layer %d wv", l);
        if (btpk_check_weight_dir(model, &lds[l].wv, what) != 0) return -1;
        snprintf(what, sizeof(what), "layer %d wo", l);
        if (btpk_check_weight_dir(model, &lds[l].wo, what) != 0) return -1;
        snprintf(what, sizeof(what), "layer %d w_gate", l);
        if (btpk_check_weight_dir(model, &lds[l].w_gate, what) != 0) return -1;
        snprintf(what, sizeof(what), "layer %d w_up", l);
        if (btpk_check_weight_dir(model, &lds[l].w_up, what) != 0) return -1;
        snprintf(what, sizeof(what), "layer %d w_down", l);
        if (btpk_check_weight_dir(model, &lds[l].w_down, what) != 0) return -1;

        LOAD_F32(attn_norm,     attn_norm_off,     cfg->dim);
        LOAD_F32(attn_sub_norm, attn_sub_norm_off, cfg->dim);
        LOAD_F32(ffn_norm,      ffn_norm_off,      cfg->dim);
        LOAD_F32(ffn_sub_norm,  ffn_sub_norm_off,  cfg->ffn_dim);

        LOAD_BTPK_W(&bl->wq,     wq);
        LOAD_BTPK_W(&bl->wk,     wk);
        LOAD_BTPK_W(&bl->wv,     wv);
        LOAD_BTPK_W(&bl->wo,     wo);
        LOAD_BTPK_W(&bl->w_gate, w_gate);
        LOAD_BTPK_W(&bl->w_up,   w_up);
        LOAD_BTPK_W(&bl->w_down, w_down);

        /* Populate the i2s_weight rows/cols/scale/tmac so code paths
         * that read those fields (FPGA GEMV, result dequant, state
         * allocation) see the right dimensions. tmac stays NULL in the
         * btpk path — FPGA fast path uses model->btpk directly. */
        lw->wq.rows = bl->wq.rows; lw->wq.cols = bl->wq.cols; lw->wq.scale = bl->wq.scale;
        lw->wk.rows = bl->wk.rows; lw->wk.cols = bl->wk.cols; lw->wk.scale = bl->wk.scale;
        lw->wv.rows = bl->wv.rows; lw->wv.cols = bl->wv.cols; lw->wv.scale = bl->wv.scale;
        lw->wo.rows = bl->wo.rows; lw->wo.cols = bl->wo.cols; lw->wo.scale = bl->wo.scale;
        lw->w_gate.rows = bl->w_gate.rows; lw->w_gate.cols = bl->w_gate.cols; lw->w_gate.scale = bl->w_gate.scale;
        lw->w_up.rows   = bl->w_up.rows;   lw->w_up.cols   = bl->w_up.cols;   lw->w_up.scale   = bl->w_up.scale;
        lw->w_down.rows = bl->w_down.rows; lw->w_down.cols = bl->w_down.cols; lw->w_down.scale = bl->w_down.scale;
    }

    #undef LOAD_F32
    #undef LOAD_BTPK_W

    fprintf(stderr, "biturbo: .btpk loaded (%.1f MB mmap'd, zero repack)\n",
            (double)model->mmap_size / (1024 * 1024));
    return 0;
}
#endif /* BT_FPGA */

int bt_load_model(bt_model_t* model, const char* path) {
    memset(model, 0, sizeof(*model));

    /* mmap the file */
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "biturbo: cannot open '%s'\n", path);
        return -1;
    }
    struct stat st;
    fstat(fd, &st);
    model->mmap_size = (size_t)st.st_size;
    model->mmap_data = (uint8_t*)mmap(NULL, model->mmap_size,
                                       PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (model->mmap_data == MAP_FAILED) {
        fprintf(stderr, "biturbo: mmap failed\n");
        return -1;
    }

    /* Dispatch on magic: "BTPKMDL\0" → pre-packed fast path */
    if (model->mmap_size >= 8 &&
        memcmp(model->mmap_data, "BTPKMDL\0", 8) == 0) {
#ifndef BT_FPGA
        fprintf(stderr, "biturbo: .btpk requires a BT_FPGA build — "
                "use the GGUF model for CPU inference\n");
        munmap(model->mmap_data, model->mmap_size);
        model->mmap_data = NULL;
        return -1;
#else
        if (bt_load_btpk(model) != 0) {
            bt_free_model(model);
            return -1;
        }
#endif
#ifdef BT_FPGA
        {
            uint32_t ddr3_base = 0x24000000;
            uint32_t ddr3_avm_base = 0x24000000;
            uint32_t ddr3_span = 0x1C000000;
            const char *s;
            if ((s = getenv("BT_FPGA_DDR3_BASE")) != NULL)
                ddr3_base = (uint32_t)strtoul(s, NULL, 0);
            if ((s = getenv("BT_FPGA_DDR3_AVM_BASE")) != NULL)
                ddr3_avm_base = (uint32_t)strtoul(s, NULL, 0);
            if ((s = getenv("BT_FPGA_DDR3_SPAN")) != NULL)
                ddr3_span = (uint32_t)strtoul(s, NULL, 0);
            if (bt_fpga_init(ddr3_base, ddr3_avm_base, ddr3_span) == 0) {
                if (bt_fpga_prepare_layout_btpk(model) != 0) {
                    fprintf(stderr,
                            "biturbo: FPGA layout init failed; .btpk requires FPGA fast path\n");
                    bt_free_model(model);
                    return -1;
                }
                fprintf(stderr,
                        "biturbo: FPGA T-MAC ready (btpk fast path, backend=%s, layout=%s, persistent_weights=%s)\n",
                        bt_fpga_backend_name(bt_fpga_mem_backend),
                        bt_fpga_layout_name(bt_fpga_layout),
                        bt_fpga_layout == BT_FPGA_LAYOUT_PERSISTENT ? "yes" : "no");
            } else {
                fprintf(stderr,
                        "biturbo: FPGA init failed; .btpk requires FPGA fast path\n");
                bt_free_model(model);
                return -1;
            }
        }
#endif
        return 0;
    }

    gguf_reader_t r = { model->mmap_data, 0, model->mmap_size };

    /* Header */
    uint32_t magic = rd_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "biturbo: not a GGUF or .btpk file\n");
        munmap(model->mmap_data, model->mmap_size);
        return -1;
    }
    uint32_t version = rd_u32(&r);
    if (version < 2 || version > 3) {
        fprintf(stderr, "biturbo: unsupported GGUF version %u\n", version);
        munmap(model->mmap_data, model->mmap_size);
        return -1;
    }
    uint64_t n_tensors = rd_u64(&r);
    uint64_t n_kv = rd_u64(&r);

    fprintf(stderr, "biturbo: GGUF v%u — %llu tensors, %llu metadata\n",
            version, (unsigned long long)n_tensors, (unsigned long long)n_kv);

    /* Parse metadata → extract config and tokenizer */
    bt_config_t* cfg = &model->config;
    cfg->max_seq_len = 4096;  /* default */

    bt_tokenizer_t* tok = &model->tokenizer;
    tok->bos_id = -1;
    tok->eos_id = -1;
    tok->eot_id = -1;

    /* We need to do two passes or save positions for arrays.
     * For token arrays, save reader position and parse later. */
    size_t tok_tokens_pos = 0;
    uint64_t tok_tokens_count = 0;
    size_t tok_scores_pos = 0;
    uint64_t tok_scores_count = 0;
    (void)0; /* tok_types: reserved for future use */

    for (uint64_t i = 0; i < n_kv; i++) {
        gguf_str_t key = rd_str(&r);
        uint32_t vtype = rd_u32(&r);

        /* Model config */
        if (str_eq(key, "bitnet-b1.58.embedding_length") && vtype == GGUF_TYPE_U32) {
            cfg->dim = (int)rd_u32(&r);
        } else if (str_eq(key, "bitnet-b1.58.block_count") && vtype == GGUF_TYPE_U32) {
            cfg->n_layers = (int)rd_u32(&r);
        } else if (str_eq(key, "bitnet-b1.58.attention.head_count") && vtype == GGUF_TYPE_U32) {
            cfg->n_heads = (int)rd_u32(&r);
        } else if (str_eq(key, "bitnet-b1.58.attention.head_count_kv") && vtype == GGUF_TYPE_U32) {
            cfg->n_kv_heads = (int)rd_u32(&r);
        } else if (str_eq(key, "bitnet-b1.58.vocab_size") && vtype == GGUF_TYPE_U32) {
            cfg->vocab_size = (int)rd_u32(&r);
        } else if (str_eq(key, "bitnet-b1.58.feed_forward_length") && vtype == GGUF_TYPE_U32) {
            cfg->ffn_dim = (int)rd_u32(&r);
        } else if (str_eq(key, "bitnet-b1.58.context_length") && vtype == GGUF_TYPE_U32) {
            cfg->max_seq_len = (int)rd_u32(&r);
        } else if (str_eq(key, "bitnet-b1.58.attention.layer_norm_rms_epsilon") && vtype == GGUF_TYPE_F32) {
            cfg->norm_eps = rd_f32(&r);
        } else if (str_eq(key, "bitnet-b1.58.rope.freq_base") && vtype == GGUF_TYPE_F32) {
            cfg->rope_theta = rd_f32(&r);
        } else if (str_eq(key, "bitnet-b1.58.rope.freq_base") && vtype == GGUF_TYPE_F64) {
            cfg->rope_theta = (float)rd_f64(&r);
        /* Tokenizer arrays: save positions for later */
        } else if (str_eq(key, "tokenizer.ggml.tokens") && vtype == GGUF_TYPE_ARR) {
            uint32_t etype = rd_u32(&r);
            tok_tokens_count = rd_u64(&r);
            tok_tokens_pos = r.pos;
            /* Skip the string array */
            if (etype == GGUF_TYPE_STR) {
                for (uint64_t j = 0; j < tok_tokens_count; j++) rd_str(&r);
            }
        } else if (str_eq(key, "tokenizer.ggml.scores") && vtype == GGUF_TYPE_ARR) {
            uint32_t etype = rd_u32(&r);
            tok_scores_count = rd_u64(&r);
            tok_scores_pos = r.pos;
            r.pos += tok_scores_count * type_size(etype);
        } else if (str_eq(key, "tokenizer.ggml.bos_token_id") && vtype == GGUF_TYPE_U32) {
            tok->bos_id = (int)rd_u32(&r);
        } else if (str_eq(key, "tokenizer.ggml.eos_token_id") && vtype == GGUF_TYPE_U32) {
            tok->eos_id = (int)rd_u32(&r);
        } else {
            skip_val(&r, vtype);
        }
    }

    fprintf(stderr, "biturbo: dim=%d layers=%d heads=%d kv_heads=%d "
            "vocab=%d ffn=%d ctx=%d\n",
            cfg->dim, cfg->n_layers, cfg->n_heads, cfg->n_kv_heads,
            cfg->vocab_size, cfg->ffn_dim, cfg->max_seq_len);

    /* Build tokenizer from saved positions */
    if (tok_tokens_count > 0) {
        tok->vocab_size = (int)tok_tokens_count;
        tok->vocab = (char**)bt_calloc(tok->vocab_size, sizeof(char*));
        tok->scores = (float*)bt_calloc(tok->vocab_size, sizeof(float));
        tok->max_token_len = 0;

        /* Read token strings */
        gguf_reader_t tr = { model->mmap_data, tok_tokens_pos, model->mmap_size };
        for (int i = 0; i < tok->vocab_size; i++) {
            gguf_str_t s = rd_str(&tr);
            tok->vocab[i] = (char*)bt_calloc(s.len + 1, 1);
            memcpy(tok->vocab[i], s.str, s.len);
            tok->vocab[i][s.len] = '\0';
            if ((int)s.len > tok->max_token_len)
                tok->max_token_len = (int)s.len;
        }

        /* Read scores */
        if (tok_scores_count > 0) {
            gguf_reader_t sr = { model->mmap_data, tok_scores_pos, model->mmap_size };
            for (int i = 0; i < tok->vocab_size && i < (int)tok_scores_count; i++) {
                tok->scores[i] = rd_f32(&sr);
            }
        }

        /* Find special tokens by content if not in metadata */
        if (tok->bos_id < 0) {
            for (int i = 0; i < tok->vocab_size; i++) {
                if (strcmp(tok->vocab[i], "<|begin_of_text|>") == 0 ||
                    strcmp(tok->vocab[i], "<s>") == 0) {
                    tok->bos_id = i; break;
                }
            }
        }
        if (tok->eos_id < 0) {
            for (int i = 0; i < tok->vocab_size; i++) {
                if (strcmp(tok->vocab[i], "<|end_of_text|>") == 0 ||
                    strcmp(tok->vocab[i], "</s>") == 0) {
                    tok->eos_id = i; break;
                }
            }
        }
        for (int i = 0; i < tok->vocab_size; i++) {
            if (strcmp(tok->vocab[i], "<|eot_id|>") == 0) {
                tok->eot_id = i; break;
            }
        }

        /* Build sorted vocab index for O(log V) binary search */
        tok->sorted_idx = (int*)bt_calloc(tok->vocab_size, sizeof(int));
        for (int i = 0; i < tok->vocab_size; i++) tok->sorted_idx[i] = i;
        bt_sort_vocab_global = tok->vocab;
        qsort(tok->sorted_idx, tok->vocab_size, sizeof(int), bt_sort_vocab_cmp);

        fprintf(stderr, "biturbo: tokenizer: %d tokens, max_len=%d, "
                "bos=%d eos=%d\n",
                tok->vocab_size, tok->max_token_len, tok->bos_id, tok->eos_id);
    }

    /* Read tensor info headers */
    gguf_tensor_info_t* tensors = NULL;
    tensors = (gguf_tensor_info_t*)bt_calloc(
        n_tensors, sizeof(gguf_tensor_info_t));
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_str_t name = rd_str(&r);
        tensors[i].name = name.str;
        tensors[i].name_len = (int)name.len;
        tensors[i].n_dims = rd_u32(&r);
        for (uint32_t d = 0; d < tensors[i].n_dims; d++)
            tensors[i].dims[d] = rd_u64(&r);
        tensors[i].type = rd_u32(&r);
        tensors[i].offset = rd_u64(&r);
    }

    /* Data section starts at aligned offset */
    size_t alignment = 32;
    size_t data_offset = r.pos;
    data_offset += (alignment - data_offset % alignment) % alignment;
    const uint8_t* data_base = model->mmap_data + data_offset;

    /* Allocate state */
    alloc_state(&model->state, cfg);

    /* Load weights */
    bt_weights_t* wt = &model->weights;
    int n_t = (int)n_tensors;

    /* Token embedding (Q6_K, mmap'd directly) */
    const gguf_tensor_info_t* te = find_tensor(tensors, n_t, "token_embd.weight");
    if (!te) { fprintf(stderr, "biturbo: missing token_embd.weight\n"); goto fail; }
    wt->token_embedding = (const uint8_t*)(data_base + te->offset);

    /* Final norm */
    const gguf_tensor_info_t* fn = find_tensor(tensors, n_t, "output_norm.weight");
    if (!fn) { fprintf(stderr, "biturbo: missing output_norm.weight\n"); goto fail; }
    wt->final_norm = load_f32_tensor(fn, data_base);

    /* Per-layer weights */
    wt->layers = (bt_layer_weights_t*)bt_calloc(cfg->n_layers,
                                                 sizeof(bt_layer_weights_t));
    for (int l = 0; l < cfg->n_layers; l++) {
        bt_layer_weights_t* lw = &wt->layers[l];
        char nm[128];
        const gguf_tensor_info_t* ti;

        /* Helper: find tensor or fail */
        #define MUST_FIND(pattern) do { \
            snprintf(nm, sizeof(nm), pattern, l); \
            ti = find_tensor(tensors, n_t, nm); \
            if (!ti) { fprintf(stderr, "biturbo: missing %s\n", nm); goto fail; } \
        } while(0)

        MUST_FIND("blk.%d.attn_norm.weight");
        lw->attn_norm = load_f32_tensor(ti, data_base);

        MUST_FIND("blk.%d.attn_q.weight");
        init_i2s_weight(&lw->wq, ti, data_base);
        MUST_FIND("blk.%d.attn_k.weight");
        init_i2s_weight(&lw->wk, ti, data_base);
        MUST_FIND("blk.%d.attn_v.weight");
        init_i2s_weight(&lw->wv, ti, data_base);

        MUST_FIND("blk.%d.attn_sub_norm.weight");
        lw->attn_sub_norm = load_f32_tensor(ti, data_base);

        MUST_FIND("blk.%d.attn_output.weight");
        init_i2s_weight(&lw->wo, ti, data_base);

        MUST_FIND("blk.%d.ffn_norm.weight");
        lw->ffn_norm = load_f32_tensor(ti, data_base);

        MUST_FIND("blk.%d.ffn_gate.weight");
        init_i2s_weight(&lw->w_gate, ti, data_base);
        MUST_FIND("blk.%d.ffn_up.weight");
        init_i2s_weight(&lw->w_up, ti, data_base);

        MUST_FIND("blk.%d.ffn_sub_norm.weight");
        lw->ffn_sub_norm = load_f32_tensor(ti, data_base);

        MUST_FIND("blk.%d.ffn_down.weight");
        init_i2s_weight(&lw->w_down, ti, data_base);

        #undef MUST_FIND
    }

    free(tensors);

    /* Repack all I2_S weights into T-MAC TL2 format */
    for (int l = 0; l < cfg->n_layers; l++) {
        bt_layer_weights_t* lw = &wt->layers[l];
        tmac_repack(&lw->wq);
        tmac_repack(&lw->wk);
        tmac_repack(&lw->wv);
        tmac_repack(&lw->wo);
        tmac_repack(&lw->w_gate);
        tmac_repack(&lw->w_up);
        tmac_repack(&lw->w_down);
    }
    fprintf(stderr, "biturbo: T-MAC TL2 weights repacked (%d layers)\n",
            cfg->n_layers);

#ifdef BT_FPGA
    {
        uint32_t ddr3_base = 0x24000000;
        uint32_t ddr3_avm_base = 0x24000000;
        uint32_t ddr3_span = 0x1C000000;
        const char *s;
        if ((s = getenv("BT_FPGA_DDR3_BASE")) != NULL)
            ddr3_base = (uint32_t)strtoul(s, NULL, 0);
        if ((s = getenv("BT_FPGA_DDR3_AVM_BASE")) != NULL)
            ddr3_avm_base = (uint32_t)strtoul(s, NULL, 0);
        if ((s = getenv("BT_FPGA_DDR3_SPAN")) != NULL)
            ddr3_span = (uint32_t)strtoul(s, NULL, 0);
        if (bt_fpga_init(ddr3_base, ddr3_avm_base, ddr3_span) == 0) {
            if (bt_fpga_prepare_layout(model) == 0) {
                fprintf(stderr,
                        "biturbo: FPGA T-MAC accelerator ready (backend=%s, layout=%s, persistent_weights=%s)\n",
                        bt_fpga_backend_name(bt_fpga_mem_backend),
                        bt_fpga_layout_name(bt_fpga_layout),
                        bt_fpga_layout == BT_FPGA_LAYOUT_PERSISTENT ? "yes" : "no");
            } else {
                fprintf(stderr, "biturbo: FPGA layout init failed, using CPU path\n");
                bt_fpga_cleanup();
            }
        } else {
            fprintf(stderr, "biturbo: FPGA init failed, using CPU path\n");
        }
    }
#endif

    fprintf(stderr, "biturbo: model loaded (%.1f MB mmap'd)\n",
            (double)model->mmap_size / (1024 * 1024));
    return 0;

fail:
    free(tensors);
    bt_free_model(model);
    return -1;
}

void bt_free_model(bt_model_t* model) {
#ifdef BT_FPGA
    if (bt_fpga_regs)
        bt_fpga_cleanup();
#endif
    /* Free allocated weight copies (F32 norms) */
    bt_weights_t* wt = &model->weights;
    if (wt->layers) {
        for (int l = 0; l < model->config.n_layers; l++) {
            bt_layer_weights_t* lw = &wt->layers[l];
            free(lw->attn_norm);
            free(lw->attn_sub_norm);
            free(lw->ffn_norm);
            free(lw->ffn_sub_norm);
            tmac_free(&lw->wq);
            tmac_free(&lw->wk);
            tmac_free(&lw->wv);
            tmac_free(&lw->wo);
            tmac_free(&lw->w_gate);
            tmac_free(&lw->w_up);
            tmac_free(&lw->w_down);
        }
        free(wt->layers);
    }
    free(wt->final_norm);

    /* Free state */
    bt_state_t* s = &model->state;
    free(s->x); free(s->xb); free(s->xb2);
    free(s->hb); free(s->hb2);
    free(s->q); free(s->k); free(s->v);
    free(s->att); free(s->q8_buf); free(s->lut_buf);
    free(s->q_rht); free(s->q_qjl); free(s->logits);
    if (s->kv) {
        for (int l = 0; l < model->config.n_layers; l++) {
            free(s->kv[l].k_cache);
            free(s->kv[l].v_cache);
        }
        free(s->kv);
    }

    /* Free tokenizer */
    bt_tokenizer_t* tok = &model->tokenizer;
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->vocab[i]);
        free(tok->vocab);
    }
    free(tok->scores);
    free(tok->sorted_idx);

    /* Free btpk directory (nib/sign data lives in mmap; nothing to free) */
    if (model->btpk) {
        free(model->btpk->layers);
        free(model->btpk);
    }

    /* Unmap */
    if (model->mmap_data && model->mmap_data != MAP_FAILED)
        munmap(model->mmap_data, model->mmap_size);

    memset(model, 0, sizeof(*model));
}

/* ================================================================
 * §12. TEXT GENERATION
 * ================================================================ */

void bt_generate(bt_model_t* model, bt_sampler_t* sampler,
                 const char* prompt, int max_tokens) {
    bt_tokenizer_t* tok = &model->tokenizer;
    bt_state_t* s = &model->state;

    int* prompt_tokens = (int*)bt_calloc(max_tokens, sizeof(int));
    int n_prompt = bt_encode(tok, prompt, prompt_tokens, max_tokens, 1);
    if (n_prompt <= 0) {
        fprintf(stderr, "biturbo: failed to encode prompt\n");
        free(prompt_tokens);
        return;
    }
    fprintf(stderr, "biturbo: %d prompt tokens\n", n_prompt);

    int token = prompt_tokens[0];
    int prev_token = 0;
    struct timespec ts_start;
    int gen_count = 0;
    double layers_total = 0.0;
    double layer_stage_totals[BT_PROFILE_LAYER_STAGE_COUNT] = {0};
    double qkv_stage_totals[BT_PROFILE_QKV_STAGE_COUNT] = {0};
    double lm_head_total = 0.0;
    double sampling_total = 0.0;

    for (int pos = 0; pos < max_tokens; pos++) {
        int generating = (pos >= n_prompt - 1);
        if (generating && gen_count == 0)
            clock_gettime(CLOCK_MONOTONIC, &ts_start);

        bt_forward(model, token, pos);
        if (generating) {
            for (int i = 0; i < BT_PROFILE_LAYER_STAGE_COUNT; i++)
                layer_stage_totals[i] += s->profile_last_layer_stage_sec[i];
            for (int i = 0; i < BT_PROFILE_QKV_STAGE_COUNT; i++)
                qkv_stage_totals[i] += s->profile_last_qkv_stage_sec[i];
            layers_total += s->profile_last_layers_sec;
            lm_head_total += s->profile_last_lm_head_sec;
        }

        int next;
        if (pos < n_prompt - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            struct timespec ts_sample_start, ts_sample_end;
            clock_gettime(CLOCK_MONOTONIC, &ts_sample_start);
            next = bt_sample(sampler, model->state.logits,
                             model->config.vocab_size);
            clock_gettime(CLOCK_MONOTONIC, &ts_sample_end);
            sampling_total +=
                bt_timespec_elapsed_sec(ts_sample_start, ts_sample_end);
            gen_count++;
        }

        if (next == tok->eos_id || next == tok->eot_id) break;

        if (pos >= n_prompt - 1) {
            const char* piece = bt_decode(tok, prev_token, next);
            printf("%s", piece);
            fflush(stdout);
        }

        prev_token = token;
        token = next;
    }
    printf("\n");

    if (gen_count > 1) {
        struct timespec ts_end;
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        double elapsed = bt_timespec_elapsed_sec(ts_start, ts_end);
        fprintf(stderr, "biturbo: %d tokens in %.2fs (%.1f tok/s)\n",
                gen_count, elapsed, (double)gen_count / elapsed);
        fprintf(stderr, "biturbo: profile\n");
#ifdef BT_FPGA
        if (bt_fpga_regs) {
            fprintf(stderr,
                    "  runtime: build=fpga | backend=%s | persistent_weights=%s\n",
                    bt_fpga_backend_name(bt_fpga_mem_backend),
                    bt_fpga_mem_backend == BT_FPGA_MEM_CMA ? "yes" : "no");
        } else {
            fprintf(stderr, "  runtime: build=fpga | backend=cpu-fallback\n");
        }
#else
        fprintf(stderr, "  runtime: build=cpu\n");
#endif
        fprintf(stderr,
                "  generated tokens: layers=%.2fs (%.2f s/token) | "
                "lm_head=%.2fs (%.2f s/token) | sampling=%.2fs (%.4f s/token)\n",
                layers_total, layers_total / (double)gen_count,
                lm_head_total, lm_head_total / (double)gen_count,
                sampling_total, sampling_total / (double)gen_count);
        bt_profile_print_layer_stage_summary(layer_stage_totals, gen_count,
                                             model->config.n_layers);
        bt_profile_print_qkv_stage_summary(qkv_stage_totals, gen_count,
                                           model->config.n_layers);
    }
    free(prompt_tokens);
}
