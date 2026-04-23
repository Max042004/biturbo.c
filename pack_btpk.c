/*
 * pack_btpk.c — offline pre-packer: GGUF → .btpk
 *
 * Loads a BitNet GGUF via the regular biturbo loader (which runs
 * tmac_repack), then engine-stripes each ternary weight into the
 * exact byte image the DE10-Nano T-MAC accelerator expects in DDR3
 * and writes everything to a standalone .btpk file.
 *
 * At inference time, bt_load_model() recognizes .btpk and each layer
 * switch becomes a flat memcpy into DDR3 — no nibble/sign shuffling
 * on the ARM hot path.
 *
 * Usage: pack_btpk <model.gguf> <out.btpk>
 */

#include "biturbo.h"
#include "biturbo_btpk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* T-MAC FPGA constants (must match biturbo_fpga.h) */
#define BTPK_NUM_ENGINES 32
#define BTPK_BEAT_BYTES  16

/* bt_tmac_weight_t struct is fully defined in biturbo.h. */

/* ================================================================
 * Engine-striping: lift bt_fpga_repack_to_ddr3 to emit into a buffer
 * ================================================================ */

static void btpk_weight_sizes(int rows, int cols,
                              int* k_padded_out, int* n3_padded_out,
                              int* nib_stride_out, int* sign_stride_out,
                              size_t* nib_size_out, size_t* sign_size_out) {
    int k_padded = ((cols + 2) / 3) * 3;
    int n3 = k_padded / 3;
    int tiles_per_row = (n3 + BTPK_NUM_ENGINES - 1) / BTPK_NUM_ENGINES;
    int sign_beats_per_row = (n3 + 128 - 1) / 128;

    *k_padded_out = k_padded;
    *n3_padded_out = n3;
    *nib_stride_out = tiles_per_row * BTPK_BEAT_BYTES;
    *sign_stride_out = sign_beats_per_row * BTPK_BEAT_BYTES;
    *nib_size_out  = (size_t)rows * (*nib_stride_out);
    *sign_size_out = (size_t)rows * (*sign_stride_out);
}

/*
 * Mirror of bt_fpga_repack_to_ddr3(): transforms CPU-format T-MAC
 * nibbles/signs (packed 2/byte, packed 8/byte) into engine-striped
 * DDR3 layout — each 128-bit beat carries one nibble per engine.
 *
 * Input: tw->three_nib (rows × tw->nib3_stride), three_sign (rows × sign_stride).
 *        tw->n3 is the CPU (unpadded) three-group count = cols/3.
 * Output: nib_out (rows × nib_stride), sign_out (rows × sign_stride).
 *        Positions beyond tw->n3 are left zero ⇒ ternary 0, matching
 *        FPGA's k-padding behaviour.
 */
static void stripe_to_fpga(const struct bt_tmac_weight* tw,
                           uint8_t* nib_out, int nib_stride,
                           uint8_t* sign_out, int sign_stride) {
    int rows = tw->rows;
    int n3_cpu = tw->n3;
    int n3_total = (tw->cols + 2) / 3;
    int tiles_per_row = nib_stride / BTPK_BEAT_BYTES;

    memset(nib_out,  0, (size_t)rows * nib_stride);
    memset(sign_out, 0, (size_t)rows * sign_stride);

    for (int r = 0; r < rows; r++) {
        const uint8_t* nib_row  = tw->three_nib  + (size_t)r * tw->nib3_stride;
        const uint8_t* sign_row = tw->three_sign + (size_t)r * tw->sign_stride;
        uint8_t* nib_row_out  = nib_out  + (size_t)r * nib_stride;
        uint8_t* sign_row_out = sign_out + (size_t)r * sign_stride;

        for (int t = 0; t < tiles_per_row; t++) {
            uint8_t* beat = nib_row_out + t * BTPK_BEAT_BYTES;
            for (int e = 0; e < BTPK_NUM_ENGINES; e++) {
                int g = t * BTPK_NUM_ENGINES + e;
                int nibble = 0;
                int sign_bit = 0;

                if (g >= n3_total) break;

                if (g < n3_cpu) {
                    uint8_t packed = nib_row[g / 2];
                    nibble = (g & 1) ? (packed & 0x0F) : (packed >> 4);
                    sign_bit = (sign_row[g / 8] >> (g & 7)) & 1;
                } else {
                    int tail_rc = bt_tmac_tail_group_encode(tw, r, &nibble, &sign_bit);
                    if (tail_rc < 0) {
                        nibble = 0;
                        sign_bit = 0;
                    }
                }

                {
                    int byte_pos = e / 2;
                    if (e & 1) beat[byte_pos] |= (uint8_t)(nibble << 4);
                    else       beat[byte_pos] |= (uint8_t)(nibble & 0x0F);
                }

                if (sign_bit) {
                    int tile = g / BTPK_NUM_ENGINES;
                    int eng  = g % BTPK_NUM_ENGINES;
                    int sign_beat = tile / 4;
                    int chunk = tile % 4;
                    int byte_off = sign_beat * BTPK_BEAT_BYTES + chunk * 4 + eng / 8;
                    sign_row_out[byte_off] |= (uint8_t)(1 << (eng & 7));
                }
            }
        }
    }
}

/* ================================================================
 * File I/O helpers
 * ================================================================ */

static void die(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vfprintf(stderr, fmt, ap); va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}

static uint64_t align_up(uint64_t off, uint64_t a) {
    return (off + a - 1) & ~(a - 1);
}

static void fwrite_at(FILE* f, uint64_t off, const void* data, size_t n) {
    if (fseek(f, (long)off, SEEK_SET) != 0) die("fseek(%llu) failed", (unsigned long long)off);
    if (fwrite(data, 1, n, f) != n) die("fwrite(%zu) failed", n);
}

static void fzero_to(FILE* f, uint64_t off) {
    long cur = ftell(f);
    if (cur < 0) die("ftell failed");
    if ((uint64_t)cur < off) {
        static const uint8_t zeros[4096] = {0};
        uint64_t need = off - (uint64_t)cur;
        while (need > 0) {
            size_t chunk = need > sizeof(zeros) ? sizeof(zeros) : (size_t)need;
            if (fwrite(zeros, 1, chunk, f) != chunk) die("fwrite(zeros) failed");
            need -= chunk;
        }
    } else if ((uint64_t)cur > off) {
        die("internal: file cursor %ld past expected offset %llu",
            cur, (unsigned long long)off);
    }
}

/* ================================================================
 * Weight serialization
 * ================================================================ */

typedef struct {
    uint64_t* cursor;
    FILE*     out;
} emit_ctx_t;

/* Forward declarations from biturbo.c internals we access via casts */

static void emit_weight(const bt_i2s_weight_t* w,
                        btpk_weight_dir_t* dir,
                        emit_ctx_t* ctx) {
    const struct bt_tmac_weight* tw =
        (const struct bt_tmac_weight*)w->tmac;
    if (!tw) die("weight has NULL tmac — was tmac_repack run?");

    int k_padded, n3_padded, nib_stride, sign_stride;
    size_t nib_size, sign_size;
    btpk_weight_sizes(tw->rows, tw->cols,
                      &k_padded, &n3_padded,
                      &nib_stride, &sign_stride,
                      &nib_size, &sign_size);

    uint8_t* nib  = (uint8_t*)calloc(1, nib_size);
    uint8_t* sign = (uint8_t*)calloc(1, sign_size);
    if (!nib || !sign) die("OOM striping weight %dx%d", tw->rows, tw->cols);

    stripe_to_fpga(tw, nib, nib_stride, sign, sign_stride);

    uint64_t nib_off  = align_up(*ctx->cursor, BTPK_BLOB_ALIGN);
    fzero_to(ctx->out, nib_off);
    fwrite_at(ctx->out, nib_off, nib, nib_size);
    *ctx->cursor = nib_off + nib_size;

    uint64_t sign_off = align_up(*ctx->cursor, BTPK_BLOB_ALIGN);
    fzero_to(ctx->out, sign_off);
    fwrite_at(ctx->out, sign_off, sign, sign_size);
    *ctx->cursor = sign_off + sign_size;

    free(nib);
    free(sign);

    dir->rows        = (uint32_t)tw->rows;
    dir->cols        = (uint32_t)tw->cols;
    dir->n3          = n3_padded;
    dir->k_padded    = k_padded;
    dir->nib_stride  = nib_stride;
    dir->sign_stride = sign_stride;
    dir->nib_off     = nib_off;
    dir->nib_size    = nib_size;
    dir->sign_off    = sign_off;
    dir->sign_size   = sign_size;
    dir->scale       = tw->scale;
    dir->_pad        = 0;
}

static uint64_t emit_f32(const float* data, size_t n_floats,
                         emit_ctx_t* ctx) {
    uint64_t off = align_up(*ctx->cursor, BTPK_BLOB_ALIGN);
    fzero_to(ctx->out, off);
    fwrite_at(ctx->out, off, data, n_floats * sizeof(float));
    *ctx->cursor = off + n_floats * sizeof(float);
    return off;
}

static uint64_t emit_bytes(const void* data, size_t n_bytes,
                           emit_ctx_t* ctx) {
    uint64_t off = align_up(*ctx->cursor, BTPK_BLOB_ALIGN);
    fzero_to(ctx->out, off);
    fwrite_at(ctx->out, off, data, n_bytes);
    *ctx->cursor = off + n_bytes;
    return off;
}

/* ================================================================
 * LM head SVD sidecars (optional)
 *
 * Read the three raw binary files produced by tools/compute_lmhead_svd.py:
 *   <dir>/lm_head_V.f32       — [dim,   rank] float32
 *   <dir>/lm_head_E_q.i8      — [vocab, rank] int8
 *   <dir>/lm_head_E_scale.f32 — [vocab]       float32
 *
 * Rank is derived from the V.f32 file size (dim*rank*4 bytes), and the
 * other two files are cross-checked against (dim, vocab, rank). meta.json
 * is ignored — the file sizes are the authoritative description.
 * ================================================================ */

typedef struct {
    int      rank;
    float*   V;          /* [dim,   rank] */
    int8_t*  E_q;        /* [vocab, rank] */
    float*   E_scale;    /* [vocab]       */
    size_t   V_bytes;
    size_t   E_q_bytes;
    size_t   E_scale_bytes;
} lmhead_svd_t;

static void* slurp_file(const char* path, size_t* out_bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) die("cannot open sidecar '%s'", path);
    if (fseek(f, 0, SEEK_END) != 0) die("fseek END failed on '%s'", path);
    long n = ftell(f);
    if (n < 0) die("ftell failed on '%s'", path);
    if (fseek(f, 0, SEEK_SET) != 0) die("fseek SET failed on '%s'", path);
    void* buf = malloc((size_t)n);
    if (!buf) die("OOM slurping '%s' (%ld bytes)", path, n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n)
        die("fread short on '%s'", path);
    fclose(f);
    *out_bytes = (size_t)n;
    return buf;
}

static void load_lmhead_svd(const char* dir, int dim, int vocab,
                            lmhead_svd_t* out) {
    char path[1024];

    snprintf(path, sizeof(path), "%s/lm_head_V.f32", dir);
    out->V = (float*)slurp_file(path, &out->V_bytes);
    if (out->V_bytes % (sizeof(float) * (size_t)dim) != 0)
        die("lm_head_V.f32 size %zu is not a multiple of dim*%zu=%zu",
            out->V_bytes, sizeof(float),
            sizeof(float) * (size_t)dim);
    out->rank = (int)(out->V_bytes / (sizeof(float) * (size_t)dim));
    if (out->rank <= 0 || out->rank > dim)
        die("derived rank=%d is invalid (dim=%d)", out->rank, dim);
    fprintf(stderr, "[pack] lm_head SVD: derived rank=%d from V size\n",
            out->rank);

    snprintf(path, sizeof(path), "%s/lm_head_E_q.i8", dir);
    out->E_q = (int8_t*)slurp_file(path, &out->E_q_bytes);
    size_t want_eq = (size_t)vocab * (size_t)out->rank;
    if (out->E_q_bytes != want_eq)
        die("lm_head_E_q.i8 size %zu != vocab*rank=%zu",
            out->E_q_bytes, want_eq);

    snprintf(path, sizeof(path), "%s/lm_head_E_scale.f32", dir);
    out->E_scale = (float*)slurp_file(path, &out->E_scale_bytes);
    size_t want_sc = (size_t)vocab * sizeof(float);
    if (out->E_scale_bytes != want_sc)
        die("lm_head_E_scale.f32 size %zu != vocab*4=%zu",
            out->E_scale_bytes, want_sc);
}

static void free_lmhead_svd(lmhead_svd_t* s) {
    free(s->V); free(s->E_q); free(s->E_scale);
    memset(s, 0, sizeof(*s));
}

/* ================================================================
 * Tokenizer blob: scores + (len, bytes) per token
 * ================================================================ */

static void emit_tokenizer(const bt_tokenizer_t* tok,
                           btpk_header_t* hdr,
                           emit_ctx_t* ctx) {
    size_t est = tok->vocab_size * sizeof(float);
    for (int i = 0; i < tok->vocab_size; i++)
        est += 2 + strlen(tok->vocab[i]);

    uint8_t* buf = (uint8_t*)calloc(1, est);
    if (!buf) die("OOM tokenizer blob");

    size_t o = 0;
    memcpy(buf + o, tok->scores, tok->vocab_size * sizeof(float));
    o += tok->vocab_size * sizeof(float);

    for (int i = 0; i < tok->vocab_size; i++) {
        size_t L = strlen(tok->vocab[i]);
        if (L > 0xFFFF) die("token %d too long (%zu)", i, L);
        uint16_t len = (uint16_t)L;
        memcpy(buf + o, &len, 2); o += 2;
        memcpy(buf + o, tok->vocab[i], L); o += L;
    }

    uint64_t off = align_up(*ctx->cursor, BTPK_BLOB_ALIGN);
    fzero_to(ctx->out, off);
    fwrite_at(ctx->out, off, buf, o);
    *ctx->cursor = off + o;

    hdr->tokenizer_off     = off;
    hdr->tokenizer_size    = (uint64_t)o;
    hdr->tok_vocab_size    = tok->vocab_size;
    hdr->tok_max_token_len = tok->max_token_len;
    hdr->tok_bos_id        = tok->bos_id;
    hdr->tok_eos_id        = tok->eos_id;
    hdr->tok_eot_id        = tok->eot_id;
    hdr->_tok_pad          = 0;

    free(buf);
}

/* ================================================================
 * Q6_K token embedding blob — copied verbatim from mmap'd GGUF
 * ================================================================ */

#define BTPK_QK_K 256
typedef struct {
    uint8_t  ql[BTPK_QK_K/2];
    uint8_t  qh[BTPK_QK_K/4];
    int8_t   scales[BTPK_QK_K/16];
    uint16_t d;
} btpk_q6k_block_t;

static void emit_token_embed(const uint8_t* token_embedding,
                             const bt_config_t* cfg,
                             btpk_header_t* hdr,
                             emit_ctx_t* ctx) {
    int nb_per_row = cfg->dim / BTPK_QK_K;
    size_t row_bytes = (size_t)nb_per_row * sizeof(btpk_q6k_block_t);
    size_t total = (size_t)cfg->vocab_size * row_bytes;

    uint64_t off = align_up(*ctx->cursor, BTPK_BLOB_ALIGN);
    fzero_to(ctx->out, off);
    fwrite_at(ctx->out, off, token_embedding, total);
    *ctx->cursor = off + total;

    hdr->token_embed_off  = off;
    hdr->token_embed_size = total;
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char** argv) {
    const char* gguf_path = NULL;
    const char* out_path  = NULL;
    const char* lmhead_svd_dir = NULL;

    /* Parse: <gguf> <out.btpk> [--lmhead-svd-dir <dir>] */
    int pos_count = 0;
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (strcmp(a, "--lmhead-svd-dir") == 0) {
            if (i + 1 >= argc)
                die("--lmhead-svd-dir requires a path argument");
            lmhead_svd_dir = argv[++i];
        } else if (a[0] == '-') {
            die("unknown flag: %s", a);
        } else {
            if (pos_count == 0) gguf_path = a;
            else if (pos_count == 1) out_path = a;
            else die("too many positional arguments");
            pos_count++;
        }
    }
    if (pos_count != 2) {
        fprintf(stderr,
                "Usage: %s <model.gguf> <out.btpk> "
                "[--lmhead-svd-dir <sidecar_dir>]\n",
                argv[0]);
        return 1;
    }

    bt_model_t model;
    if (bt_load_model(&model, gguf_path) != 0)
        die("failed to load GGUF '%s'", gguf_path);
    fprintf(stderr, "[pack] GGUF loaded; tmac_repack done\n");

    FILE* out = fopen(out_path, "wb+");
    if (!out) die("cannot open '%s' for write", out_path);

    /* Reserve header + layer-dir area; fill in later. */
    btpk_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic[0] = BTPK_MAGIC0; hdr.magic[1] = BTPK_MAGIC1;
    hdr.magic[2] = BTPK_MAGIC2; hdr.magic[3] = BTPK_MAGIC3;
    hdr.magic[4] = BTPK_MAGIC4; hdr.magic[5] = BTPK_MAGIC5;
    hdr.magic[6] = BTPK_MAGIC6; hdr.magic[7] = BTPK_MAGIC7;
    hdr.version     = BTPK_VERSION;
    hdr.format      = BTPK_FMT_TMAC_TL2;
    hdr.num_engines = BTPK_NUM_ENGINES;
    hdr.beat_bytes  = BTPK_BEAT_BYTES;
    hdr.dim         = model.config.dim;
    hdr.n_layers    = model.config.n_layers;
    hdr.n_heads     = model.config.n_heads;
    hdr.n_kv_heads  = model.config.n_kv_heads;
    hdr.vocab_size  = model.config.vocab_size;
    hdr.ffn_dim     = model.config.ffn_dim;
    hdr.max_seq_len = model.config.max_seq_len;
    hdr.norm_eps    = model.config.norm_eps;
    hdr.rope_theta  = model.config.rope_theta;

    btpk_layer_dir_t* layers = (btpk_layer_dir_t*)
        calloc(model.config.n_layers, sizeof(btpk_layer_dir_t));
    if (!layers) die("OOM layer dirs");

    uint64_t cursor = sizeof(btpk_header_t);
    /* Reserve layer-dir area now so everything else appends after. */
    uint64_t layers_off = align_up(cursor, BTPK_BLOB_ALIGN);
    cursor = layers_off + (uint64_t)model.config.n_layers * sizeof(btpk_layer_dir_t);
    hdr.layers_off = layers_off;

    emit_ctx_t ctx = { &cursor, out };

    /* Tokenizer */
    emit_tokenizer(&model.tokenizer, &hdr, &ctx);
    fprintf(stderr, "[pack] tokenizer: %zu bytes\n", (size_t)hdr.tokenizer_size);

    /* Token embedding */
    emit_token_embed(model.weights.token_embedding, &model.config, &hdr, &ctx);
    fprintf(stderr, "[pack] token_embed: %zu bytes\n", (size_t)hdr.token_embed_size);

    /* Final norm */
    hdr.final_norm_off  = emit_f32(model.weights.final_norm,
                                   (size_t)model.config.dim, &ctx);
    hdr.final_norm_size = (uint64_t)model.config.dim * sizeof(float);

    /* LM head SVD factors (optional). Read sidecars and emit three sections;
     * leaves hdr.lm_head_rank = 0 untouched when no directory is given. */
    lmhead_svd_t svd; memset(&svd, 0, sizeof(svd));
    if (lmhead_svd_dir) {
        load_lmhead_svd(lmhead_svd_dir,
                        model.config.dim, model.config.vocab_size, &svd);

        size_t V_bytes    = (size_t)model.config.dim * svd.rank * sizeof(float);
        size_t E_q_bytes  = (size_t)model.config.vocab_size * svd.rank;
        size_t E_sc_bytes = (size_t)model.config.vocab_size * sizeof(float);

        hdr.lm_head_rank         = svd.rank;
        hdr.lm_head_V_off        = emit_bytes(svd.V,       V_bytes,    &ctx);
        hdr.lm_head_V_size       = V_bytes;
        hdr.lm_head_E_q_off      = emit_bytes(svd.E_q,     E_q_bytes,  &ctx);
        hdr.lm_head_E_q_size     = E_q_bytes;
        hdr.lm_head_E_scale_off  = emit_bytes(svd.E_scale, E_sc_bytes, &ctx);
        hdr.lm_head_E_scale_size = E_sc_bytes;

        fprintf(stderr,
                "[pack] lm_head SVD: rank=%d, V=%.2f MB, E_q=%.2f MB, "
                "E_scale=%.2f MB\n",
                svd.rank, V_bytes / 1e6, E_q_bytes / 1e6, E_sc_bytes / 1e6);
        free_lmhead_svd(&svd);
    }

    /* Per-layer */
    for (int l = 0; l < model.config.n_layers; l++) {
        bt_layer_weights_t* lw = &model.weights.layers[l];
        btpk_layer_dir_t*   ld = &layers[l];

        ld->attn_norm_off     = emit_f32(lw->attn_norm,     model.config.dim,     &ctx);
        ld->attn_sub_norm_off = emit_f32(lw->attn_sub_norm, model.config.dim,     &ctx);
        ld->ffn_norm_off      = emit_f32(lw->ffn_norm,      model.config.dim,     &ctx);
        ld->ffn_sub_norm_off  = emit_f32(lw->ffn_sub_norm,  model.config.ffn_dim, &ctx);

        emit_weight(&lw->wq,     &ld->wq,     &ctx);
        emit_weight(&lw->wk,     &ld->wk,     &ctx);
        emit_weight(&lw->wv,     &ld->wv,     &ctx);
        emit_weight(&lw->wo,     &ld->wo,     &ctx);
        emit_weight(&lw->w_gate, &ld->w_gate, &ctx);
        emit_weight(&lw->w_up,   &ld->w_up,   &ctx);
        emit_weight(&lw->w_down, &ld->w_down, &ctx);

        if ((l & 0x3) == 0)
            fprintf(stderr, "[pack] layer %d/%d done\n", l+1, model.config.n_layers);
    }

    hdr.total_file_size = cursor;

    /* Backfill header + layer dirs */
    fwrite_at(out, 0, &hdr, sizeof(hdr));
    fwrite_at(out, layers_off, layers,
              (size_t)model.config.n_layers * sizeof(btpk_layer_dir_t));

    /* Ensure the file is exactly total_file_size bytes on disk. */
    if (fseek(out, 0, SEEK_END) != 0) die("fseek END failed");
    long end = ftell(out);
    if (end < 0) die("ftell failed");
    if ((uint64_t)end != hdr.total_file_size) {
        /* Rare: last blob was shorter than a cursor bump would suggest,
         * or we had to zero-pad. Truncate or extend to match. */
        if (ftruncate(fileno(out), (off_t)hdr.total_file_size) != 0)
            die("ftruncate to %llu failed",
                (unsigned long long)hdr.total_file_size);
    }

    fclose(out);
    bt_free_model(&model);

    fprintf(stderr, "[pack] wrote %s (%.1f MB)\n",
            out_path, (double)hdr.total_file_size / (1024.0 * 1024.0));
    free(layers);
    return 0;
}
