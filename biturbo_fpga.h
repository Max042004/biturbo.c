/*
 * biturbo_fpga.h -- T-MAC FPGA accelerator driver for biturbo.c
 *
 * Offloads T-MAC GEMV to DE10-Nano FPGA (TMacAccelerator).
 * The FPGA builds LUTs from INT8 activations internally, streams
 * T-MAC weights from DDR3, and returns INT32 accumulator results.
 *
 * Zero extra CPU memory: weights are repacked directly from software
 * tmac format to DDR3 on each layer switch (~14 MB memcpy per layer).
 */

#ifndef BITURBO_FPGA_H
#define BITURBO_FPGA_H

#include "biturbo.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>

/* ================================================================
 * T-MAC FPGA register map
 * ================================================================ */

#define BT_FPGA_LW_BRIDGE_BASE  0xFF200000
#define BT_FPGA_LW_BRIDGE_SPAN  0x00200000  /* 2 MB */

#define BT_REG_CTRL           0x00
#define BT_REG_STATUS         0x04
#define BT_REG_NIB_BASE       0x08
#define BT_REG_DIM_M          0x0C
#define BT_REG_DIM_K          0x10
#define BT_REG_SIGN_BASE      0x14
#define BT_REG_PERF_CYCLES    0x18
#define BT_REG_DIM_N3         0x1C  /* HPS-supplied K/3 to avoid on-chip divider */
#define BT_REG_ACT_DDR3_BASE  0x28
#define BT_REG_RES_DDR3_BASE  0x2C

#define BT_CTRL_START         0x01
#define BT_CTRL_DDR3_MODE     0x02

#define BT_FPGA_MAX_DIM_M     8192
#define BT_FPGA_MAX_DIM_K     8192
#define BT_FPGA_NUM_ENGINES   32    /* parallel lookup engines */
#define BT_FPGA_BEAT_BYTES    16    /* 128-bit DDR3 bus = 16 bytes */

/* ================================================================
 * Global state
 * ================================================================ */

static int               bt_fpga_fd = -1;
static volatile uint32_t *bt_fpga_regs = NULL;
static volatile uint8_t  *bt_fpga_ddr3 = NULL;
static uint32_t          bt_fpga_ddr3_phys = 0;
static uint32_t          bt_fpga_ddr3_span = 0;
static uint32_t          bt_fpga_act_off = 0;
static uint32_t          bt_fpga_res_off = 0;
static int               bt_fpga_cached_layer = -1;
static int               bt_fpga_timeout_us = 2000000;
static int32_t          *bt_fpga_raw_buf = NULL;

/* Per-weight DDR3 offset */
typedef struct {
    uint32_t nib_off;
    uint32_t sign_off;
    int      nib_stride;   /* bytes per row in nibble array */
    int      sign_stride;  /* bytes per row in sign array */
} bt_fpga_wt_loc_t;

typedef struct {
    bt_fpga_wt_loc_t wq, wk, wv, wo;
    bt_fpga_wt_loc_t w_gate, w_up, w_down;
} bt_fpga_layer_loc_t;

static bt_fpga_layer_loc_t *bt_fpga_layer_locs = NULL;
static int bt_fpga_n_layers = 0;

/* ================================================================
 * Low-level register access
 * ================================================================ */

static inline void bt_fpga_reg_write(uint32_t offset, uint32_t val) {
    ((volatile uint32_t *)bt_fpga_regs)[offset / 4] = val;
}

static inline uint32_t bt_fpga_reg_read(uint32_t offset) {
    return ((volatile uint32_t *)bt_fpga_regs)[offset / 4];
}

/* ================================================================
 * Init / Cleanup
 * ================================================================ */

static int bt_fpga_init(uint32_t ddr3_base, uint32_t ddr3_span) {
    bt_fpga_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (bt_fpga_fd < 0) {
        perror("bt_fpga_init: open /dev/mem");
        return -1;
    }

    volatile void *lw = mmap(NULL, BT_FPGA_LW_BRIDGE_SPAN,
                             PROT_READ | PROT_WRITE, MAP_SHARED,
                             bt_fpga_fd, BT_FPGA_LW_BRIDGE_BASE);
    if (lw == MAP_FAILED) {
        perror("bt_fpga_init: mmap lw_bridge");
        close(bt_fpga_fd);
        return -1;
    }
    bt_fpga_regs = (volatile uint32_t *)lw;

    bt_fpga_ddr3 = (volatile uint8_t *)mmap(NULL, ddr3_span,
                    PROT_READ | PROT_WRITE, MAP_SHARED,
                    bt_fpga_fd, ddr3_base);
    if (bt_fpga_ddr3 == MAP_FAILED) {
        perror("bt_fpga_init: mmap ddr3");
        munmap((void *)lw, BT_FPGA_LW_BRIDGE_SPAN);
        close(bt_fpga_fd);
        return -1;
    }

    bt_fpga_ddr3_phys = ddr3_base;
    bt_fpga_ddr3_span = ddr3_span;
    bt_fpga_cached_layer = -1;

    fprintf(stderr, "[FPGA] T-MAC accelerator bound: DDR3 0x%08X span 0x%X\n",
            ddr3_base, ddr3_span);
    return 0;
}

static void bt_fpga_cleanup(void) {
    free(bt_fpga_raw_buf);
    free(bt_fpga_layer_locs);
    if (bt_fpga_ddr3 && bt_fpga_ddr3 != MAP_FAILED)
        munmap((void *)bt_fpga_ddr3, bt_fpga_ddr3_span);
    if (bt_fpga_regs && bt_fpga_regs != MAP_FAILED)
        munmap((void *)bt_fpga_regs, BT_FPGA_LW_BRIDGE_SPAN);
    if (bt_fpga_fd >= 0)
        close(bt_fpga_fd);
    bt_fpga_fd = -1;
    bt_fpga_regs = NULL;
    bt_fpga_ddr3 = NULL;
    bt_fpga_raw_buf = NULL;
    bt_fpga_layer_locs = NULL;
    bt_fpga_cached_layer = -1;
}

/* ================================================================
 * Wait for DONE
 * ================================================================ */

static int bt_fpga_wait_done(void) {
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (unsigned spins = 0; ; spins++) {
        uint32_t st = bt_fpga_reg_read(BT_REG_STATUS);
        if (st & 0x2) return 0;

        if ((spins & 0xFF) == 0) {
            clock_gettime(CLOCK_MONOTONIC, &now);
            long elapsed = (long)(now.tv_sec - start.tv_sec) * 1000000L +
                           (long)(now.tv_nsec - start.tv_nsec) / 1000L;
            if (elapsed >= bt_fpga_timeout_us) {
                fprintf(stderr, "[FPGA] timeout: STATUS=0x%08X PERF=%u\n",
                        st, bt_fpga_reg_read(BT_REG_PERF_CYCLES));
                return -1;
            }
        }
    }
}

/* Dump all registers for debugging */
static void bt_fpga_dump_regs(const char *tag) {
    fprintf(stderr, "[FPGA] %s:\n", tag);
    fprintf(stderr, "  STATUS     = 0x%08X\n", bt_fpga_reg_read(BT_REG_STATUS));
    fprintf(stderr, "  NIB_BASE   = 0x%08X\n", bt_fpga_reg_read(BT_REG_NIB_BASE));
    fprintf(stderr, "  DIM_M      = %u\n",     bt_fpga_reg_read(BT_REG_DIM_M));
    fprintf(stderr, "  DIM_K      = %u\n",     bt_fpga_reg_read(BT_REG_DIM_K));
    fprintf(stderr, "  SIGN_BASE  = 0x%08X\n", bt_fpga_reg_read(BT_REG_SIGN_BASE));
    fprintf(stderr, "  PERF_CYCLES= %u\n",     bt_fpga_reg_read(BT_REG_PERF_CYCLES));
    fprintf(stderr, "  ACT_BASE   = 0x%08X\n", bt_fpga_reg_read(BT_REG_ACT_DDR3_BASE));
    fprintf(stderr, "  RES_BASE   = 0x%08X\n", bt_fpga_reg_read(BT_REG_RES_DDR3_BASE));
}

/* ================================================================
 * Compute DDR3 layout sizes for a weight matrix (no allocation)
 * ================================================================ */

static void bt_fpga_weight_sizes(const bt_tmac_weight_t *tw,
                                 size_t *nib_size, size_t *sign_size,
                                 int *nib_stride, int *sign_stride) {
    int k_padded = ((tw->cols + 2) / 3) * 3;
    int n3 = k_padded / 3;
    int tiles_per_row = (n3 + BT_FPGA_NUM_ENGINES - 1) / BT_FPGA_NUM_ENGINES;
    int sign_beats_per_row = (n3 + 128 - 1) / 128;

    *nib_stride = tiles_per_row * BT_FPGA_BEAT_BYTES;
    *sign_stride = sign_beats_per_row * BT_FPGA_BEAT_BYTES;
    *nib_size = (size_t)tw->rows * (*nib_stride);
    *sign_size = (size_t)tw->rows * (*sign_stride);
}

/* ================================================================
 * Compute DDR3 offsets for all layers (just arithmetic, no alloc)
 * ================================================================ */

static void bt_fpga_prepare_layout(bt_model_t *model) {
    int n_layers = model->config.n_layers;
    bt_fpga_n_layers = n_layers;
    bt_fpga_layer_locs = (bt_fpga_layer_loc_t *)calloc(n_layers,
                           sizeof(bt_fpga_layer_loc_t));

    int max_k = model->config.dim;
    if (model->config.ffn_dim > max_k) max_k = model->config.ffn_dim;
    int max_k_padded = ((max_k + 2) / 3) * 3;
    int max_m = model->config.ffn_dim;
    if (model->config.dim > max_m) max_m = model->config.dim;

    size_t max_layer_size = 0;

    for (int l = 0; l < n_layers; l++) {
        bt_layer_weights_t *lw = &model->weights.layers[l];
        bt_tmac_weight_t *wts[] = {
            lw->wq.tmac, lw->wk.tmac, lw->wv.tmac, lw->wo.tmac,
            lw->w_gate.tmac, lw->w_up.tmac, lw->w_down.tmac
        };
        bt_fpga_wt_loc_t *locs[] = {
            &bt_fpga_layer_locs[l].wq, &bt_fpga_layer_locs[l].wk,
            &bt_fpga_layer_locs[l].wv, &bt_fpga_layer_locs[l].wo,
            &bt_fpga_layer_locs[l].w_gate, &bt_fpga_layer_locs[l].w_up,
            &bt_fpga_layer_locs[l].w_down
        };

        uint32_t offset = 0;
        for (int i = 0; i < 7; i++) {
            size_t nsz, ssz;
            int ns, ss;
            bt_fpga_weight_sizes(wts[i], &nsz, &ssz, &ns, &ss);

            offset = (offset + BT_FPGA_BEAT_BYTES - 1) & ~(BT_FPGA_BEAT_BYTES - 1);
            locs[i]->nib_off = offset;
            locs[i]->nib_stride = ns;
            offset += (uint32_t)nsz;

            offset = (offset + BT_FPGA_BEAT_BYTES - 1) & ~(BT_FPGA_BEAT_BYTES - 1);
            locs[i]->sign_off = offset;
            locs[i]->sign_stride = ss;
            offset += (uint32_t)ssz;
        }

        if (offset > max_layer_size)
            max_layer_size = offset;
    }

    bt_fpga_act_off = (uint32_t)((max_layer_size + 4095) & ~4095UL);
    bt_fpga_res_off = bt_fpga_act_off + (uint32_t)((max_k_padded + 4095) & ~4095UL);
    uint32_t total_needed = bt_fpga_res_off + (uint32_t)max_m * 4 + 4096;

    if (total_needed > bt_fpga_ddr3_span) {
        fprintf(stderr, "[FPGA] WARNING: need %u bytes but DDR3 span is %u\n",
                total_needed, bt_fpga_ddr3_span);
    }

    bt_fpga_raw_buf = (int32_t *)malloc((size_t)max_m * sizeof(int32_t));

    fprintf(stderr, "[FPGA] DDR3 layout: weights=%zu, act@0x%X, res@0x%X, total=%u\n",
            max_layer_size, bt_fpga_act_off, bt_fpga_res_off, total_needed);
}

/* ================================================================
 * Repack one weight matrix directly from software tmac → DDR3
 *
 * Writes directly to mmap'd DDR3 region. No CPU-side allocation.
 * ================================================================ */

static void bt_fpga_repack_to_ddr3(const bt_tmac_weight_t *tw,
                                   const bt_fpga_wt_loc_t *loc) {
    int rows = tw->rows;
    int n3 = tw->n3;
    int nib_stride = loc->nib_stride;
    int sign_stride = loc->sign_stride;

    uint8_t *nib_base = (uint8_t *)bt_fpga_ddr3 + loc->nib_off;
    uint8_t *sign_base = (uint8_t *)bt_fpga_ddr3 + loc->sign_off;

    /* Zero the destination regions */
    memset(nib_base, 0, (size_t)rows * nib_stride);
    memset(sign_base, 0, (size_t)rows * sign_stride);

    for (int r = 0; r < rows; r++) {
        const uint8_t *nib_row = tw->three_nib + (size_t)r * tw->nib3_stride;
        const uint8_t *sign_row = tw->three_sign + (size_t)r * tw->sign_stride;
        uint8_t *nib_out = nib_base + (size_t)r * nib_stride;
        uint8_t *sign_out = sign_base + (size_t)r * sign_stride;

        /* Repack nibbles: software (even g=high nib) → FPGA (even engine=low nib) */
        int tiles_per_row = nib_stride / BT_FPGA_BEAT_BYTES;
        for (int t = 0; t < tiles_per_row; t++) {
            uint8_t *beat = nib_out + t * BT_FPGA_BEAT_BYTES;
            for (int e = 0; e < BT_FPGA_NUM_ENGINES; e++) {
                int g = t * BT_FPGA_NUM_ENGINES + e;
                if (g >= n3) break;

                uint8_t packed = nib_row[g / 2];
                int nibble = (g & 1) ? (packed & 0x0F) : (packed >> 4);

                int byte_pos = e / 2;
                if (e & 1)
                    beat[byte_pos] |= (uint8_t)(nibble << 4);
                else
                    beat[byte_pos] |= (uint8_t)(nibble & 0x0F);
            }
        }

        /* Repack signs into 128-bit beats (4 tiles × 32 sign bits each) */
        for (int g = 0; g < n3; g++) {
            int sign_bit = (sign_row[g / 8] >> (g & 7)) & 1;
            if (!sign_bit) continue;

            int tile = g / BT_FPGA_NUM_ENGINES;
            int eng = g % BT_FPGA_NUM_ENGINES;
            int sign_beat = tile / 4;
            int chunk = tile % 4;

            int byte_off = sign_beat * BT_FPGA_BEAT_BYTES + chunk * 4 + eng / 8;
            sign_out[byte_off] |= (uint8_t)(1 << (eng & 7));
        }
    }
}

/* ================================================================
 * Load one layer's weights to DDR3 (repack on-the-fly, with caching)
 * ================================================================ */

static void bt_fpga_load_layer(bt_model_t *model, int layer) {
    if (bt_fpga_cached_layer == layer) return;

    bt_layer_weights_t *lw = &model->weights.layers[layer];
    bt_tmac_weight_t *wts[] = {
        lw->wq.tmac, lw->wk.tmac, lw->wv.tmac, lw->wo.tmac,
        lw->w_gate.tmac, lw->w_up.tmac, lw->w_down.tmac
    };
    bt_fpga_wt_loc_t *locs[] = {
        &bt_fpga_layer_locs[layer].wq, &bt_fpga_layer_locs[layer].wk,
        &bt_fpga_layer_locs[layer].wv, &bt_fpga_layer_locs[layer].wo,
        &bt_fpga_layer_locs[layer].w_gate, &bt_fpga_layer_locs[layer].w_up,
        &bt_fpga_layer_locs[layer].w_down
    };

    for (int i = 0; i < 7; i++)
        bt_fpga_repack_to_ddr3(wts[i], locs[i]);

    bt_fpga_cached_layer = layer;
}

/* ================================================================
 * Upload INT8 activations to DDR3 (pad K to multiple of 3)
 * ================================================================ */

static void bt_fpga_upload_activations(const int8_t *q8_buf, int K) {
    int k_padded = ((K + 2) / 3) * 3;
    uint8_t *dst = (uint8_t *)bt_fpga_ddr3 + bt_fpga_act_off;
    memcpy(dst, q8_buf, (size_t)K);
    for (int i = K; i < k_padded; i++)
        dst[i] = 0;
}

/* ================================================================
 * FPGA GEMV: run one T-MAC matrix-vector multiply
 * ================================================================ */

static int bt_fpga_gemv_count = 0;

static void bt_fpga_gemv(const bt_tmac_weight_t *tw,
                         const bt_fpga_wt_loc_t *loc,
                         int32_t *results) {
    int M = tw->rows;
    int K = ((tw->cols + 2) / 3) * 3;

    uint32_t nib_phys = bt_fpga_ddr3_phys + loc->nib_off;
    uint32_t sign_phys = bt_fpga_ddr3_phys + loc->sign_off;
    uint32_t act_phys = bt_fpga_ddr3_phys + bt_fpga_act_off;
    uint32_t res_phys = bt_fpga_ddr3_phys + bt_fpga_res_off;

    int rows_done = 0;
    while (rows_done < M) {
        int tile_m = M - rows_done;
        if (tile_m > BT_FPGA_MAX_DIM_M)
            tile_m = BT_FPGA_MAX_DIM_M;

        uint32_t tile_nib = nib_phys + (uint32_t)rows_done * loc->nib_stride;
        uint32_t tile_sign = sign_phys + (uint32_t)rows_done * loc->sign_stride;

        /* Debug: dump registers before first GEMV */
        if (bt_fpga_gemv_count == 0) {
            fprintf(stderr, "[FPGA] first GEMV: M=%d K=%d\n", M, K);
            fprintf(stderr, "  NIB_BASE  =0x%08X\n", tile_nib);
            fprintf(stderr, "  SIGN_BASE =0x%08X\n", tile_sign);
            fprintf(stderr, "  ACT_BASE  =0x%08X\n", act_phys);
            fprintf(stderr, "  RES_BASE  =0x%08X\n", res_phys);
            fprintf(stderr, "  STATUS before START=0x%08X\n",
                    bt_fpga_reg_read(BT_REG_STATUS));
        }

        bt_fpga_reg_write(BT_REG_NIB_BASE, tile_nib);
        bt_fpga_reg_write(BT_REG_SIGN_BASE, tile_sign);
        bt_fpga_reg_write(BT_REG_DIM_M, (uint32_t)tile_m);
        bt_fpga_reg_write(BT_REG_DIM_K, (uint32_t)K);
        bt_fpga_reg_write(BT_REG_DIM_N3, (uint32_t)(K / 3));
        bt_fpga_reg_write(BT_REG_ACT_DDR3_BASE, act_phys);
        bt_fpga_reg_write(BT_REG_RES_DDR3_BASE, res_phys);
        bt_fpga_reg_write(BT_REG_CTRL, BT_CTRL_START | BT_CTRL_DDR3_MODE);

        if (bt_fpga_gemv_count == 0) {
            /* Read back immediately after START */
            fprintf(stderr, "  STATUS after START =0x%08X\n",
                    bt_fpga_reg_read(BT_REG_STATUS));
            /* Readback registers to verify writes took effect */
            fprintf(stderr, "  Readback NIB_BASE  =0x%08X\n",
                    bt_fpga_reg_read(BT_REG_NIB_BASE));
            fprintf(stderr, "  Readback DIM_M     =%u\n",
                    bt_fpga_reg_read(BT_REG_DIM_M));
            fprintf(stderr, "  Readback DIM_K     =%u\n",
                    bt_fpga_reg_read(BT_REG_DIM_K));
            fprintf(stderr, "  Readback SIGN_BASE =0x%08X\n",
                    bt_fpga_reg_read(BT_REG_SIGN_BASE));
        }
        bt_fpga_gemv_count++;

        if (bt_fpga_wait_done() < 0) {
            fprintf(stderr, "[FPGA] GEMV timeout M=%d K=%d tile_offset=%d\n",
                    M, K, rows_done);
            memset(&results[rows_done], 0, (size_t)tile_m * sizeof(int32_t));
            rows_done += tile_m;
            continue;
        }

        memcpy(&results[rows_done],
               (void *)(bt_fpga_ddr3 + bt_fpga_res_off),
               (size_t)tile_m * sizeof(int32_t));
        rows_done += tile_m;
    }
}

/* ================================================================
 * FPGA-accelerated GEMV with dequantization
 * ================================================================ */

static void bt_fpga_gemv_dequant(float *out, const bt_tmac_weight_t *tw,
                                 const bt_fpga_wt_loc_t *loc,
                                 float inv_scale) {
    bt_fpga_gemv(tw, loc, bt_fpga_raw_buf);
    float dequant = inv_scale * tw->scale;
    for (int r = 0; r < tw->rows; r++)
        out[r] = (float)bt_fpga_raw_buf[r] * dequant;
}

#endif /* BITURBO_FPGA_H */
