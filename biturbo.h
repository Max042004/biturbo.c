/*
 * biturbo.h — Zero-dependency BitNet 1.58-bit inference engine with TurboQuant KV cache
 *
 * Architecture: BitNet-b1.58 transformer with:
 *   - 1.58-bit ternary weights {-1, 0, +1} in GGUF I2_S format
 *   - Dynamic per-token INT8 activation quantization (BitLinear)
 *   - TurboQuant uniform INT4 KV cache quantization
 *   - Grouped-Query Attention (GQA)
 *   - RoPE positional encoding
 *   - SqReLU-gated FFN
 *   - RMS normalization
 *
 * Loads GGUF models directly (e.g. ggml-model-i2_s.gguf from Microsoft BitNet)
 */

#ifndef BITURBO_H
#define BITURBO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Model configuration (populated from GGUF metadata)
 * ============================================================ */

typedef struct {
    int dim;            /* hidden dimension (e.g. 2560)          */
    int n_layers;       /* number of transformer blocks (e.g. 30)*/
    int n_heads;        /* number of query heads (e.g. 20)       */
    int n_kv_heads;     /* number of KV heads for GQA (e.g. 5)   */
    int vocab_size;     /* vocabulary size (e.g. 128256)         */
    int ffn_dim;        /* FFN hidden dimension (e.g. 6912)      */
    int max_seq_len;    /* maximum sequence length (e.g. 4096)   */
    float norm_eps;     /* RMS norm epsilon (e.g. 1e-5)          */
    float rope_theta;   /* RoPE frequency base (e.g. 500000.0)  */
} bt_config_t;

#define BT_HEAD_DIM(cfg) ((cfg)->dim / (cfg)->n_heads)
#define BT_KV_DIM(cfg)   ((cfg)->n_kv_heads * BT_HEAD_DIM(cfg))
#define BT_GQA_GROUPS(cfg) ((cfg)->n_heads / (cfg)->n_kv_heads)

/* ============================================================
 * I2_S ternary weight — pointer into mmap'd GGUF data
 *
 * Group-interleaved 2-bit packing (blocks of 128):
 *   byte[gp] bits: [c0:6-7] [c1:4-5] [c2:2-3] [c3:0-1]
 *   c0 → element at gp + 0*32
 *   c1 → element at gp + 1*32
 *   c2 → element at gp + 2*32
 *   c3 → element at gp + 3*32
 * Encoding: 0=-1, 1=0, 2=+1, 3=0 (unused)
 * Per-tensor float32 scale appended after packed bytes.
 * ============================================================ */

typedef struct bt_tmac_weight bt_tmac_weight_t;

typedef struct {
    const uint8_t* data;    /* pointer into mmap'd packed I2_S data       */
    float          scale;   /* per-tensor scale (from end of I2_S blob)   */
    int            rows;    /* ne1: output features (number of rows)      */
    int            cols;    /* ne0: input features (elements per row)     */
    bt_tmac_weight_t* tmac; /* T-MAC repacked weight (NULL if not repacked) */
} bt_i2s_weight_t;

/* ============================================================
 * T-MAC TL2 repacked weight
 *
 * Packs 3 ternary weights into 4-bit LUT index + 1-bit sign.
 * 3^3=27 combos, sign-paired → 14 unsigned classes in 4-bit nibble.
 * Remainder columns (K%3) use 2-weight groups (3^2=9, no sign).
 * Row-major layout: nibbles packed 2/byte, signs packed 8/byte.
 * ============================================================ */

struct bt_tmac_weight {
    uint8_t* three_nib;     /* packed 4-bit indices for 3-weight groups   */
    uint8_t* three_sign;    /* packed sign bits for 3-weight groups       */
    uint8_t* two_nib;       /* packed 4-bit indices for 2-weight groups   */
    float    scale;         /* per-tensor weight scale                    */
    int      rows;          /* output features                            */
    int      cols;          /* input features                             */
    int      n3;            /* number of 3-weight groups (cols / 3)       */
    int      n2;            /* number of 2-weight groups (0 or 1)         */
    int      nib3_stride;   /* bytes per row for three_nib                */
    int      sign_stride;   /* bytes per row for three_sign               */
    int      nib2_stride;   /* bytes per row for two_nib                  */
};

/* Convert the CPU-side two-weight remainder encoding into the equivalent
 * padded 3-weight T-MAC group (w0, w1, 0). This lets the FPGA path preserve
 * cols % 3 != 0 semantics by storing the tail as one final 3-weight group. */
static inline int bt_tmac_tail_group_encode(const bt_tmac_weight_t* tw, int row,
                                            int* nibble_out, int* sign_out) {
    static const uint8_t BT_TMAC2_TO_TMAC3_NIB[9]  = { 0, 2, 2, 3, 6, 9, 3, 9, 6 };
    static const uint8_t BT_TMAC2_TO_TMAC3_SIGN[9] = { 0, 0, 1, 0, 0, 0, 1, 1, 1 };

    if (nibble_out) *nibble_out = 0;
    if (sign_out) *sign_out = 0;

    if (!tw || tw->n2 <= 0 || !tw->two_nib) {
        return 0;
    }

    {
        const uint8_t packed = tw->two_nib[(size_t)row * tw->nib2_stride];
        const uint8_t two_nib = packed >> 4;  /* the remainder is always stored in the high nibble */
        if (two_nib >= 9) {
            return -1;
        }
        if (nibble_out) *nibble_out = (int)BT_TMAC2_TO_TMAC3_NIB[two_nib];
        if (sign_out) *sign_out = (int)BT_TMAC2_TO_TMAC3_SIGN[two_nib];
    }

    return 1;
}

/* ============================================================
 * TurboQuant KV cache block (4-bit: 3-bit codebook + 1-bit QJL)
 *
 * Pipeline: normalize → RHT → Lloyd-Max codebook → QJL residual
 * Attention: RHT(query) once → codebook dot + QJL XNOR correction
 * V dequant: codebook lookup → inverse RHT → scale by norm
 * ============================================================ */

#define BT_QK 128  /* block size (elements per block) */

typedef struct {
    uint16_t norm;                      /* L2 norm of original vector (fp16) */
    uint16_t residual_norm;             /* L2 norm of quantization residual  */
    uint32_t rht_seed;                  /* RHT random seed                   */
    uint8_t  mse_indices[BT_QK*3/8];   /* 3-bit packed codebook idx (48B)   */
    uint8_t  qjl_signs[BT_QK/8];       /* 1-bit QJL sign hash (16B)         */
} bt_kv_block_t;                        /* 72 bytes per 128 elements         */

/* ============================================================
 * KV cache (per layer)
 * ============================================================ */

typedef struct {
    bt_kv_block_t* k_cache;     /* [n_kv_heads][max_seq][blocks_per_head] */
    bt_kv_block_t* v_cache;
    int blocks_per_head;
} bt_kv_cache_t;

/* ============================================================
 * BPE tokenizer (extracted from GGUF metadata)
 * ============================================================ */

typedef struct {
    char**   vocab;         /* token strings, indexed by token id  */
    float*   scores;        /* merge priority scores               */
    int*     sorted_idx;    /* vocab indices sorted by string       */
    int      vocab_size;
    int      bos_id;
    int      eos_id;
    int      eot_id;        /* end of turn (chat models)           */
    int      max_token_len;
} bt_tokenizer_t;

/* ============================================================
 * Per-layer transformer weights (separate Q/K/V/O, gate/up)
 * ============================================================ */

typedef struct {
    float* attn_norm;           /* [dim] pre-attention RMS norm         */
    bt_i2s_weight_t wq;        /* [dim, dim] query projection          */
    bt_i2s_weight_t wk;        /* [kv_dim, dim] key projection         */
    bt_i2s_weight_t wv;        /* [kv_dim, dim] value projection       */
    float* attn_sub_norm;       /* [dim] post-attention sub-norm        */
    bt_i2s_weight_t wo;        /* [dim, dim] output projection         */
    float* ffn_norm;            /* [dim] pre-FFN RMS norm               */
    bt_i2s_weight_t w_gate;    /* [ffn_dim, dim] FFN gate projection   */
    bt_i2s_weight_t w_up;      /* [ffn_dim, dim] FFN up projection     */
    float* ffn_sub_norm;        /* [ffn_dim] FFN sub-norm               */
    bt_i2s_weight_t w_down;    /* [dim, ffn_dim] FFN down projection   */
} bt_layer_weights_t;

/* ============================================================
 * Full model weights
 * ============================================================ */

typedef struct {
    const uint8_t* token_embedding;   /* [vocab_size, dim] Q6_K (mmap'd) */
    bt_layer_weights_t* layers;       /* [n_layers]                     */
    float* final_norm;                /* [dim] output RMS norm          */
    /* lm_head: tied to token_embedding (transposed).
     * Optional SVD factors for a low-rank approximation of the LM head:
     *   logits ≈ E_prod · (V_r^T · h),  E_prod stored INT8 per-row.
     * NULL when not available (GGUF path or old .btpk), in which case
     * Stage 14 falls back to the full Q6_K dot product over the vocab. */
    const float*  lm_head_V;          /* [dim, rank]   f32             */
    const int8_t* lm_head_E_q;        /* [vocab, rank] int8            */
    const float*  lm_head_E_scale;    /* [vocab]       f32 per-row scale*/
    int           lm_head_rank;       /* 0 = SVD path disabled         */
} bt_weights_t;

/* ============================================================
 * Pre-packed .btpk model handle (FPGA-striped nib/sign blobs)
 *
 * Populated by bt_load_model when the file magic is "BTPKMDL\0".
 * The FPGA fast path consumes btpk_weight directly — each layer
 * switch becomes a memcpy(DDR3, mmap_base + off, size) per weight.
 * NULL when the model was loaded from GGUF.
 * ============================================================ */
typedef struct {
    int      rows;
    int      cols;
    int      n3;          /* serialized 3-weight groups          */
    int      k_padded;    /* ((cols + 2) / 3) * 3                */
    int      nib_stride;  /* FPGA row stride (bytes)             */
    int      sign_stride; /* FPGA row stride (bytes)             */
    uint64_t nib_size;
    uint64_t sign_size;
    const uint8_t* nib_data;   /* mmap pointer, pre-striped      */
    const uint8_t* sign_data;  /* mmap pointer, pre-striped      */
    float    scale;
} btpk_weight_t;

typedef struct {
    btpk_weight_t wq, wk, wv, wo, w_gate, w_up, w_down;
} btpk_layer_t;

typedef struct {
    btpk_layer_t* layers;     /* [n_layers] ternary weights      */
} bt_btpk_weights_t;

enum {
    BT_PROFILE_LAYER_STAGE_ATTN_NORM = 0,
    BT_PROFILE_LAYER_STAGE_WEIGHT_LOAD,
    BT_PROFILE_LAYER_STAGE_QKV,
    BT_PROFILE_LAYER_STAGE_ROPE,
    BT_PROFILE_LAYER_STAGE_KV_CACHE,
    BT_PROFILE_LAYER_STAGE_ATTENTION,
    BT_PROFILE_LAYER_STAGE_ATTN_OUT,
    BT_PROFILE_LAYER_STAGE_ATTN_RESIDUAL,
    BT_PROFILE_LAYER_STAGE_FFN_IN,
    BT_PROFILE_LAYER_STAGE_SQRELU,
    BT_PROFILE_LAYER_STAGE_FFN_OUT,
    BT_PROFILE_LAYER_STAGE_FFN_RESIDUAL,
    BT_PROFILE_LAYER_STAGE_COUNT
};

enum {
    BT_PROFILE_QKV_STAGE_QUANT = 0,
    BT_PROFILE_QKV_STAGE_PREP,
    BT_PROFILE_QKV_STAGE_UPLOAD,
    BT_PROFILE_QKV_STAGE_WQ,
    BT_PROFILE_QKV_STAGE_WK,
    BT_PROFILE_QKV_STAGE_WV,
    BT_PROFILE_QKV_STAGE_COUNT
};

/* ============================================================
 * Inference state
 * ============================================================ */

typedef struct {
    float* x;           /* hidden state [dim]                        */
    float* xb;          /* scratch [dim]                             */
    float* xb2;         /* scratch 2 [dim]                           */
    float* hb;          /* FFN hidden [ffn_dim]                      */
    float* hb2;         /* FFN hidden 2 [ffn_dim]                    */
    float* q;           /* query [dim]                               */
    float* k;           /* key [kv_dim]                              */
    float* v;           /* value [kv_dim]                            */
    float* att;         /* attention scores [n_heads, max_seq_len]   */
    int8_t* q8_buf;     /* quantized activations scratch             */
    int16_t* lut_buf;   /* T-MAC LUT scratch [max_k/3 * 16]         */
    float* q_rht;       /* RHT-rotated query scratch [head_dim]      */
    float* q_qjl;       /* QJL query projection scratch [head_dim]   */
    float* logits;      /* output logits [vocab_size]                */
    bt_kv_cache_t* kv;  /* [n_layers] KV cache                      */
    double profile_last_layers_sec;   /* last forward() transformer-stack time */
    double profile_last_layer_stage_sec[BT_PROFILE_LAYER_STAGE_COUNT];
    double profile_last_qkv_stage_sec[BT_PROFILE_QKV_STAGE_COUNT];
    double profile_last_lm_head_sec;  /* last forward() LM-head time           */
} bt_state_t;

/* ============================================================
 * Complete model handle
 * ============================================================ */

typedef struct {
    bt_config_t     config;
    bt_weights_t    weights;
    bt_state_t      state;
    bt_tokenizer_t  tokenizer;
    uint8_t*        mmap_data;
    size_t          mmap_size;
    /* Non-NULL only when model was loaded from a .btpk file.
     * The FPGA fast path keys off this to skip per-layer repack. */
    bt_btpk_weights_t* btpk;
} bt_model_t;

/* ============================================================
 * Sampling parameters
 * ============================================================ */

typedef struct {
    float temperature;
    float top_p;
    uint64_t rng_state;
} bt_sampler_t;

/* ============================================================
 * Public API
 * ============================================================ */

int  bt_load_model(bt_model_t* model, const char* gguf_path);
void bt_free_model(bt_model_t* model);

int  bt_encode(const bt_tokenizer_t* tok, const char* text,
               int* tokens, int max_tokens, int add_bos);
const char* bt_decode(const bt_tokenizer_t* tok, int prev_token, int token);

void bt_forward(bt_model_t* model, int token, int pos);

/* Same as bt_forward but skips the final norm + LM-head projection.
 * Use for prefill tokens whose logits will be discarded; only the last
 * prompt token needs the full bt_forward so its logits can be sampled. */
void bt_forward_prefill(bt_model_t* model, int token, int pos);

void bt_sampler_init(bt_sampler_t* s, float temperature, float top_p, uint64_t seed);
int  bt_sample(bt_sampler_t* s, const float* logits, int vocab_size);

void bt_generate(bt_model_t* model, bt_sampler_t* sampler,
                 const char* prompt, int max_tokens);

/* ============================================================
 * FP16 conversion helpers
 * ============================================================ */

static inline uint16_t bt_f32_to_f16(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t  exp  = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits.u >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static inline float bt_f16_to_f32(uint16_t h) {
    union { float f; uint32_t u; } bits;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) { bits.u = sign; return bits.f; }
    if (exp == 31) { bits.u = sign | 0x7F800000 | (mant << 13); return bits.f; }
    exp = exp - 15 + 127;
    bits.u = sign | (exp << 23) | (mant << 13);
    return bits.f;
}

#ifdef __cplusplus
}
#endif

#endif /* BITURBO_H */
