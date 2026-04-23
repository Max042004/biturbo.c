/*
 * biturbo_btpk.h — on-disk layout for pre-packed BitNet models (.btpk)
 *
 * A .btpk file is a standalone BitNet model whose ternary weights are
 * already engine-striped for the T-MAC FPGA accelerator. At inference
 * time the ARM host does a single contiguous memcpy per weight blob
 * into DDR3 — no nibble/sign shuffling on the hot path.
 *
 * The format stamps the engine-striping parameters (num_engines,
 * beat_bytes) so a file produced for one RTL version is rejected by
 * another. If the FPGA layout ever changes, bump BTPK_VERSION.
 *
 * Layout:
 *   [btpk_header]
 *   [tokenizer blob]
 *   [token_embed Q6_K blob]
 *   [final_norm F32]
 *   [array of btpk_layer_dir, one per layer]
 *   (per-layer data interleaved — norms + 7 weights' nib/sign blobs)
 *
 * All weight nib/sign blobs are BTPK_BLOB_ALIGN-aligned from the start
 * of the file so DMA into 128-bit DDR3 beats is beat-aligned.
 */

#ifndef BITURBO_BTPK_H
#define BITURBO_BTPK_H

#include <stdint.h>

#define BTPK_MAGIC0 'B'
#define BTPK_MAGIC1 'T'
#define BTPK_MAGIC2 'P'
#define BTPK_MAGIC3 'K'
#define BTPK_MAGIC4 'M'
#define BTPK_MAGIC5 'D'
#define BTPK_MAGIC6 'L'
#define BTPK_MAGIC7 '\0'

#define BTPK_VERSION      3
#define BTPK_FMT_TMAC_TL2 1   /* engine-striped T-MAC TL2 */
#define BTPK_BLOB_ALIGN   64  /* 4× beat size; generous cache-line alignment */

typedef struct {
    char     magic[8];       /* "BTPKMDL\0"                    */
    uint32_t version;        /* BTPK_VERSION                   */
    uint32_t format;         /* BTPK_FMT_TMAC_TL2              */
    uint32_t num_engines;    /* e.g. 32 — FPGA parallel lanes  */
    uint32_t beat_bytes;     /* e.g. 16 — DDR3 beat width      */

    /* bt_config_t mirror */
    int32_t  dim;
    int32_t  n_layers;
    int32_t  n_heads;
    int32_t  n_kv_heads;
    int32_t  vocab_size;
    int32_t  ffn_dim;
    int32_t  max_seq_len;
    float    norm_eps;
    float    rope_theta;

    /* Tokenizer */
    int32_t  tok_vocab_size;
    int32_t  tok_max_token_len;
    int32_t  tok_bos_id;
    int32_t  tok_eos_id;
    int32_t  tok_eot_id;
    int32_t  _tok_pad;
    uint64_t tokenizer_off;    /* file offset of tokenizer blob   */
    uint64_t tokenizer_size;   /* bytes                           */

    /* Token embedding (Q6_K rows of dim each, vocab_size rows) */
    uint64_t token_embed_off;
    uint64_t token_embed_size;

    /* Final RMS norm (dim floats) */
    uint64_t final_norm_off;
    uint64_t final_norm_size;  /* == dim * sizeof(float)          */

    /* LM head SVD factors (optional; rank=0 means absent).
     * Runtime uses these to approximate logits = E · h as a low-rank
     * product (V_r · h then E_q·h_proj), then refines the top-K with
     * an exact Q6_K dot against token_embed. When rank=0, the runtime
     * falls back to the full Q6_K LM head. */
    int32_t  lm_head_rank;
    int32_t  _lm_head_pad;
    uint64_t lm_head_V_off;        /* [dim, rank]   float32 */
    uint64_t lm_head_V_size;
    uint64_t lm_head_E_q_off;      /* [vocab, rank] int8    */
    uint64_t lm_head_E_q_size;
    uint64_t lm_head_E_scale_off;  /* [vocab]       float32 */
    uint64_t lm_head_E_scale_size;

    /* Per-layer directory array (n_layers entries) */
    uint64_t layers_off;       /* file offset of btpk_layer_dir[] */

    uint64_t total_file_size;  /* sanity check                    */
} btpk_header_t;

/*
 * Per-weight directory entry.
 *
 * The nib_blob is the exact byte image the FPGA expects at
 * DDR3[nib_base]. Same for sign_blob. Strides are in bytes per
 * output row; blob_size = rows * stride (no extra padding).
 *
 * When cols % 3 != 0, the tail 1-2 ternary weights are preserved as one
 * extra padded 3-weight group (w0, w1, 0). This matches the FPGA's K-padding
 * behaviour, because the activation uploader already pads missing inputs with
 * zero before LUT build.
 */
typedef struct {
    uint32_t rows;
    uint32_t cols;
    int32_t  n3;            /* serialized 3-weight groups         */
    int32_t  k_padded;      /* = ((cols + 2)/3) * 3               */
    int32_t  nib_stride;    /* bytes per row — FPGA layout        */
    int32_t  sign_stride;   /* bytes per row — FPGA layout        */
    uint64_t nib_off;       /* file offset (blob-aligned)         */
    uint64_t nib_size;      /* = rows * nib_stride                */
    uint64_t sign_off;      /* file offset (blob-aligned)         */
    uint64_t sign_size;     /* = rows * sign_stride               */
    float    scale;         /* per-tensor ternary scale           */
    uint32_t _pad;
} btpk_weight_dir_t;

/*
 * Per-layer directory entry: 4 F32 norm vectors + 7 weight dirs.
 */
typedef struct {
    uint64_t attn_norm_off;       /* dim floats       */
    uint64_t attn_sub_norm_off;   /* dim floats       */
    uint64_t ffn_norm_off;        /* dim floats       */
    uint64_t ffn_sub_norm_off;    /* ffn_dim floats   */

    btpk_weight_dir_t wq;
    btpk_weight_dir_t wk;
    btpk_weight_dir_t wv;
    btpk_weight_dir_t wo;
    btpk_weight_dir_t w_gate;
    btpk_weight_dir_t w_up;
    btpk_weight_dir_t w_down;
} btpk_layer_dir_t;

/*
 * Tokenizer blob layout (within [tokenizer_off, tokenizer_off+size)):
 *   [float32 scores[tok_vocab_size]]
 *   for i in 0..tok_vocab_size:
 *     [uint16 len][bytes len]
 *
 * sorted_idx and specials are rebuilt on load.
 */

#endif /* BITURBO_BTPK_H */
