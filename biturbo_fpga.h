/*
 * biturbo_fpga.h -- T-MAC FPGA accelerator support for biturbo.c
 *
 * Two memory backends are supported:
 *   1. legacy /dev/mem DDR carveout
 *   2. CMA-backed per-buffer allocation through /dev/biturbo-cma
 *
 * Two DDR3 layout modes are supported:
 *   - streaming: one small DDR window reused on each layer switch
 *   - persistent: every weight gets a stable DDR address and is loaded once
 *
 * The devmem backend can operate in either layout mode, while the CMA path
 * is always persistent because each weight blob is allocated as its own DMA
 * buffer. Once weights are resident, inference only updates FPGA base
 * registers.
 */

#ifndef BITURBO_FPGA_H
#define BITURBO_FPGA_H

#include "biturbo.h"
#include "biturbo_cma_ioctl.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <time.h>

/* ================================================================
 * T-MAC FPGA register map
 * ================================================================ */

#define BT_FPGA_LW_BRIDGE_BASE  0xFF200000
#define BT_FPGA_LW_BRIDGE_SPAN  0x00200000

#define BT_REG_CTRL           0x00
#define BT_REG_STATUS         0x04
#define BT_REG_NIB_BASE       0x08
#define BT_REG_DIM_M          0x0C
#define BT_REG_DIM_K          0x10
#define BT_REG_SIGN_BASE      0x14
#define BT_REG_PERF_CYCLES    0x18
#define BT_REG_DIM_N3         0x1C
#define BT_REG_ACT_DDR3_BASE  0x28
#define BT_REG_RES_DDR3_BASE  0x2C

#define BT_CTRL_START         0x01
#define BT_CTRL_DDR3_MODE     0x02

#define BT_FPGA_MAX_DIM_M     8192
#define BT_FPGA_MAX_DIM_K     8192
#define BT_FPGA_NUM_ENGINES   32
#define BT_FPGA_BEAT_BYTES    16
#define BT_FPGA_WEIGHTS_PER_LAYER 7

/* ================================================================
 * Backend state
 * ================================================================ */

typedef enum {
    BT_FPGA_MEM_NONE = 0,
    BT_FPGA_MEM_DEVMEM = 1,
    BT_FPGA_MEM_CMA = 2
} bt_fpga_mem_backend_t;

typedef enum {
    BT_FPGA_MEM_PREF_AUTO = 0,
    BT_FPGA_MEM_PREF_DEVMEM = 1,
    BT_FPGA_MEM_PREF_CMA = 2
} bt_fpga_mem_pref_t;

typedef enum {
    BT_FPGA_LAYOUT_NONE = 0,
    BT_FPGA_LAYOUT_STREAMING = 1,
    BT_FPGA_LAYOUT_PERSISTENT = 2
} bt_fpga_layout_t;

typedef enum {
    BT_FPGA_LAYOUT_PREF_AUTO = 0,
    BT_FPGA_LAYOUT_PREF_STREAMING = 1,
    BT_FPGA_LAYOUT_PREF_PERSISTENT = 2
} bt_fpga_layout_pref_t;

typedef struct {
    int               is_cma;
    int               handle;
    size_t            size;
    uint32_t          dma_addr;
    uint64_t          mmap_offset;
    volatile uint8_t *cpu;
} bt_fpga_dma_buf_t;

typedef struct {
    bt_fpga_dma_buf_t nib;
    bt_fpga_dma_buf_t sign;
    int               nib_stride;
    int               sign_stride;
} bt_fpga_wt_loc_t;

typedef struct {
    bt_fpga_wt_loc_t wq, wk, wv, wo;
    bt_fpga_wt_loc_t w_gate, w_up, w_down;
} bt_fpga_layer_loc_t;

static int                bt_fpga_mem_fd = -1;
static int                bt_fpga_cma_fd = -1;
static volatile uint32_t *bt_fpga_regs = NULL;
static volatile uint8_t  *bt_fpga_ddr3 = NULL;
static bt_fpga_mem_backend_t bt_fpga_mem_backend = BT_FPGA_MEM_NONE;
static bt_fpga_layout_t    bt_fpga_layout = BT_FPGA_LAYOUT_NONE;
static bt_fpga_layout_pref_t bt_fpga_layout_pref = BT_FPGA_LAYOUT_PREF_AUTO;
static uint32_t           bt_fpga_ddr3_cpu_phys = 0;
static uint32_t           bt_fpga_ddr3_avm_base = 0;
static uint32_t           bt_fpga_ddr3_span = 0;
static bt_fpga_dma_buf_t  bt_fpga_act_buf;
static bt_fpga_dma_buf_t  bt_fpga_res_buf;
static int                bt_fpga_cached_layer = -1;
static int                bt_fpga_timeout_us = 2000000;
static int32_t           *bt_fpga_raw_buf = NULL;
static bt_fpga_layer_loc_t *bt_fpga_layer_locs = NULL;
static int                bt_fpga_n_layers = 0;
static size_t             bt_fpga_weight_bytes = 0;

/* ================================================================
 * Helpers
 * ================================================================ */

static inline void bt_fpga_reg_write(uint32_t offset, uint32_t val) {
    ((volatile uint32_t *)bt_fpga_regs)[offset / 4] = val;
}

static inline uint32_t bt_fpga_reg_read(uint32_t offset) {
    return ((volatile uint32_t *)bt_fpga_regs)[offset / 4];
}

static size_t bt_fpga_align_up_size(size_t value, size_t align) {
    return (value + align - 1) & ~(align - 1);
}

static const char *bt_fpga_backend_name(bt_fpga_mem_backend_t backend) {
    switch (backend) {
    case BT_FPGA_MEM_DEVMEM: return "devmem";
    case BT_FPGA_MEM_CMA: return "cma";
    default: return "none";
    }
}

static const char *bt_fpga_layout_name(bt_fpga_layout_t layout) {
    switch (layout) {
    case BT_FPGA_LAYOUT_STREAMING: return "streaming";
    case BT_FPGA_LAYOUT_PERSISTENT: return "persistent";
    default: return "none";
    }
}

static bt_fpga_mem_pref_t bt_fpga_get_mem_pref(void) {
    const char *s = getenv("BT_FPGA_MEM_BACKEND");

    if (!s || !*s || strcmp(s, "auto") == 0)
        return BT_FPGA_MEM_PREF_AUTO;
    if (strcmp(s, "devmem") == 0 || strcmp(s, "carveout") == 0)
        return BT_FPGA_MEM_PREF_DEVMEM;
    if (strcmp(s, "cma") == 0)
        return BT_FPGA_MEM_PREF_CMA;

    fprintf(stderr,
            "[FPGA] unknown BT_FPGA_MEM_BACKEND='%s', using auto\n", s);
    return BT_FPGA_MEM_PREF_AUTO;
}

static bt_fpga_layout_pref_t bt_fpga_get_layout_pref(void) {
    const char *s = getenv("BT_FPGA_DDR_LAYOUT");

    if (!s || !*s)
        s = getenv("BT_FPGA_LAYOUT");
    if (!s || !*s)
        return BT_FPGA_LAYOUT_PREF_AUTO;
    if (strcmp(s, "auto") == 0)
        return BT_FPGA_LAYOUT_PREF_AUTO;
    if (strcmp(s, "streaming") == 0 || strcmp(s, "stream") == 0)
        return BT_FPGA_LAYOUT_PREF_STREAMING;
    if (strcmp(s, "persistent") == 0 || strcmp(s, "resident") == 0)
        return BT_FPGA_LAYOUT_PREF_PERSISTENT;

    fprintf(stderr,
            "[FPGA] unknown BT_FPGA_DDR_LAYOUT='%s', using auto\n", s);
    return BT_FPGA_LAYOUT_PREF_AUTO;
}

static int bt_fpga_dma_buf_valid(const bt_fpga_dma_buf_t *buf) {
    return buf && buf->cpu && buf->size != 0;
}

static void bt_fpga_zero_dma_buf(bt_fpga_dma_buf_t *buf) {
    if (!buf)
        return;
    memset(buf, 0, sizeof(*buf));
    buf->handle = -1;
}

static void bt_fpga_free_dma_buf(bt_fpga_dma_buf_t *buf) {
    biturbo_cma_free_req_t req;

    if (!bt_fpga_dma_buf_valid(buf))
        return;

    if (buf->is_cma) {
        munmap((void *)buf->cpu, buf->size);
        if (bt_fpga_cma_fd >= 0 && buf->handle >= 0) {
            memset(&req, 0, sizeof(req));
            req.handle = (uint32_t)buf->handle;
            if (ioctl(bt_fpga_cma_fd, BITURBO_CMA_IOCTL_FREE, &req) != 0)
                perror("[FPGA] ioctl FREE");
        }
    }

    bt_fpga_zero_dma_buf(buf);
}

static void bt_fpga_free_weight_loc(bt_fpga_wt_loc_t *loc) {
    if (!loc)
        return;
    bt_fpga_free_dma_buf(&loc->nib);
    bt_fpga_free_dma_buf(&loc->sign);
    loc->nib_stride = 0;
    loc->sign_stride = 0;
}

static void bt_fpga_free_layer_locs(void) {
    int l;

    if (!bt_fpga_layer_locs)
        return;

    for (l = 0; l < bt_fpga_n_layers; l++) {
        bt_fpga_free_weight_loc(&bt_fpga_layer_locs[l].wq);
        bt_fpga_free_weight_loc(&bt_fpga_layer_locs[l].wk);
        bt_fpga_free_weight_loc(&bt_fpga_layer_locs[l].wv);
        bt_fpga_free_weight_loc(&bt_fpga_layer_locs[l].wo);
        bt_fpga_free_weight_loc(&bt_fpga_layer_locs[l].w_gate);
        bt_fpga_free_weight_loc(&bt_fpga_layer_locs[l].w_up);
        bt_fpga_free_weight_loc(&bt_fpga_layer_locs[l].w_down);
    }

    free(bt_fpga_layer_locs);
    bt_fpga_layer_locs = NULL;
    bt_fpga_n_layers = 0;
}

static void bt_fpga_reset_layout_state(void) {
    bt_fpga_free_layer_locs();
    bt_fpga_free_dma_buf(&bt_fpga_act_buf);
    bt_fpga_free_dma_buf(&bt_fpga_res_buf);
    free(bt_fpga_raw_buf);
    bt_fpga_raw_buf = NULL;
    bt_fpga_cached_layer = -1;
    bt_fpga_layout = BT_FPGA_LAYOUT_NONE;
    bt_fpga_weight_bytes = 0;
}

static int bt_fpga_map_lw_bridge(void) {
    volatile void *lw;

    bt_fpga_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (bt_fpga_mem_fd < 0) {
        perror("bt_fpga_init: open /dev/mem");
        return -1;
    }

    lw = mmap(NULL, BT_FPGA_LW_BRIDGE_SPAN,
              PROT_READ | PROT_WRITE, MAP_SHARED,
              bt_fpga_mem_fd, BT_FPGA_LW_BRIDGE_BASE);
    if (lw == MAP_FAILED) {
        perror("bt_fpga_init: mmap lw_bridge");
        close(bt_fpga_mem_fd);
        bt_fpga_mem_fd = -1;
        return -1;
    }
    bt_fpga_regs = (volatile uint32_t *)lw;
    return 0;
}

static int bt_fpga_try_open_cma(void) {
    bt_fpga_cma_fd = open(BITURBO_CMA_DEVICE, O_RDWR | O_SYNC);
    if (bt_fpga_cma_fd < 0)
        return -1;

    bt_fpga_mem_backend = BT_FPGA_MEM_CMA;
    fprintf(stderr,
            "[FPGA] T-MAC accelerator bound: backend=%s, DMA via %s\n",
            bt_fpga_backend_name(bt_fpga_mem_backend), BITURBO_CMA_DEVICE);
    return 0;
}

static int bt_fpga_try_map_devmem(uint32_t ddr3_cpu_phys,
                                  uint32_t ddr3_avm_base,
                                  uint32_t ddr3_span) {
    bt_fpga_ddr3 = (volatile uint8_t *)mmap(NULL, ddr3_span,
                    PROT_READ | PROT_WRITE, MAP_SHARED,
                    bt_fpga_mem_fd, ddr3_cpu_phys);
    if (bt_fpga_ddr3 == MAP_FAILED) {
        perror("bt_fpga_init: mmap ddr3");
        bt_fpga_ddr3 = NULL;
        return -1;
    }

    bt_fpga_mem_backend = BT_FPGA_MEM_DEVMEM;
    bt_fpga_ddr3_cpu_phys = ddr3_cpu_phys;
    bt_fpga_ddr3_avm_base = ddr3_avm_base;
    bt_fpga_ddr3_span = ddr3_span;

    fprintf(stderr,
            "[FPGA] T-MAC accelerator bound: backend=%s, CPU DDR3 0x%08X, AVM base 0x%08X, span 0x%X\n",
            bt_fpga_backend_name(bt_fpga_mem_backend),
            ddr3_cpu_phys, ddr3_avm_base, ddr3_span);
    return 0;
}

static void bt_fpga_make_buf_name(char *dst, size_t dst_sz,
                                  int layer, const char *weight,
                                  const char *kind) {
    if (layer < 0)
        snprintf(dst, dst_sz, "%s.%s", weight, kind);
    else
        snprintf(dst, dst_sz, "l%02d.%s.%s", layer, weight, kind);
}

static int bt_fpga_alloc_cma_buf(bt_fpga_dma_buf_t *buf,
                                 size_t size,
                                 const char *name) {
    biturbo_cma_alloc_req_t req;
    void *cpu;

    if (!buf || bt_fpga_cma_fd < 0 || size == 0)
        return -1;

    memset(&req, 0, sizeof(req));
    req.size = (uint64_t)size;
    if (name)
        snprintf(req.name, sizeof(req.name), "%s", name);

    if (ioctl(bt_fpga_cma_fd, BITURBO_CMA_IOCTL_ALLOC, &req) != 0) {
        perror("[FPGA] ioctl ALLOC");
        return -1;
    }

    if (req.dma_addr > 0xFFFFFFFFULL) {
        fprintf(stderr, "[FPGA] DMA address 0x%08X does not fit 32-bit FPGA regs\n",
                (unsigned)req.dma_addr);
        goto fail;
    }

    cpu = mmap(NULL, (size_t)req.size, PROT_READ | PROT_WRITE, MAP_SHARED,
               bt_fpga_cma_fd, (off_t)req.mmap_offset);
    if (cpu == MAP_FAILED) {
        perror("[FPGA] mmap CMA buffer");
        goto fail;
    }

    bt_fpga_zero_dma_buf(buf);
    buf->is_cma = 1;
    buf->handle = (int)req.handle;
    buf->size = (size_t)req.size;
    buf->dma_addr = (uint32_t)req.dma_addr;
    buf->mmap_offset = req.mmap_offset;
    buf->cpu = (volatile uint8_t *)cpu;
    return 0;

fail:
    {
        biturbo_cma_free_req_t free_req;
        memset(&free_req, 0, sizeof(free_req));
        free_req.handle = req.handle;
        (void)ioctl(bt_fpga_cma_fd, BITURBO_CMA_IOCTL_FREE, &free_req);
    }
    return -1;
}

static void bt_fpga_assign_devmem_buf(bt_fpga_dma_buf_t *buf,
                                      uint32_t offset,
                                      size_t size) {
    bt_fpga_zero_dma_buf(buf);
    buf->is_cma = 0;
    buf->handle = -1;
    buf->size = size;
    buf->dma_addr = bt_fpga_ddr3_avm_base + offset;
    buf->cpu = bt_fpga_ddr3 + offset;
}

/* ================================================================
 * Init / cleanup
 * ================================================================ */

static int bt_fpga_init(uint32_t ddr3_cpu_phys,
                        uint32_t ddr3_avm_base,
                        uint32_t ddr3_span) {
    bt_fpga_mem_pref_t pref;

    bt_fpga_ddr3_cpu_phys = ddr3_cpu_phys;
    bt_fpga_ddr3_avm_base = ddr3_avm_base;
    bt_fpga_ddr3_span = ddr3_span;
    bt_fpga_mem_backend = BT_FPGA_MEM_NONE;
    bt_fpga_layout = BT_FPGA_LAYOUT_NONE;
    bt_fpga_layout_pref = bt_fpga_get_layout_pref();
    bt_fpga_ddr3 = NULL;
    bt_fpga_cached_layer = -1;

    if (bt_fpga_map_lw_bridge() != 0)
        return -1;

    pref = bt_fpga_get_mem_pref();
    if (pref != BT_FPGA_MEM_PREF_DEVMEM && bt_fpga_try_open_cma() == 0)
        return 0;

    if (pref == BT_FPGA_MEM_PREF_CMA) {
        fprintf(stderr,
                "[FPGA] BT_FPGA_MEM_BACKEND=cma but %s is unavailable\n",
                BITURBO_CMA_DEVICE);
        munmap((void *)bt_fpga_regs, BT_FPGA_LW_BRIDGE_SPAN);
        close(bt_fpga_mem_fd);
        bt_fpga_regs = NULL;
        bt_fpga_mem_fd = -1;
        return -1;
    }

    if (bt_fpga_try_map_devmem(ddr3_cpu_phys, ddr3_avm_base, ddr3_span) != 0) {
        munmap((void *)bt_fpga_regs, BT_FPGA_LW_BRIDGE_SPAN);
        close(bt_fpga_mem_fd);
        bt_fpga_regs = NULL;
        bt_fpga_mem_fd = -1;
        return -1;
    }

    return 0;
}

static void bt_fpga_cleanup(void) {
    bt_fpga_reset_layout_state();

    if (bt_fpga_ddr3)
        munmap((void *)bt_fpga_ddr3, bt_fpga_ddr3_span);
    if (bt_fpga_regs)
        munmap((void *)bt_fpga_regs, BT_FPGA_LW_BRIDGE_SPAN);
    if (bt_fpga_cma_fd >= 0)
        close(bt_fpga_cma_fd);
    if (bt_fpga_mem_fd >= 0)
        close(bt_fpga_mem_fd);

    bt_fpga_mem_fd = -1;
    bt_fpga_cma_fd = -1;
    bt_fpga_regs = NULL;
    bt_fpga_ddr3 = NULL;
    bt_fpga_mem_backend = BT_FPGA_MEM_NONE;
}

/* ================================================================
 * Wait for DONE
 * ================================================================ */

static int bt_fpga_wait_done(void) {
    struct timespec start, now;

    clock_gettime(CLOCK_MONOTONIC, &start);
    for ( ; ; ) {
        uint32_t st = bt_fpga_reg_read(BT_REG_STATUS);
        if (st & 0x2)
            return 0;

        clock_gettime(CLOCK_MONOTONIC, &now);
        if (((long)(now.tv_sec - start.tv_sec) * 1000000L) +
            ((long)(now.tv_nsec - start.tv_nsec) / 1000L) >= bt_fpga_timeout_us) {
            fprintf(stderr, "[FPGA] timeout: STATUS=0x%08X PERF=%u\n",
                    st, bt_fpga_reg_read(BT_REG_PERF_CYCLES));
            return -1;
        }
    }
}

/* ================================================================
 * Layout helpers
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
    *nib_size = (size_t)tw->rows * (size_t)(*nib_stride);
    *sign_size = (size_t)tw->rows * (size_t)(*sign_stride);
}

static void bt_fpga_model_limits(const bt_model_t *model,
                                 int *max_k_padded_out,
                                 int *max_m_out) {
    int max_k = model->config.dim;
    int max_m = model->config.ffn_dim;

    if (model->config.ffn_dim > max_k)
        max_k = model->config.ffn_dim;
    if (model->config.dim > max_m)
        max_m = model->config.dim;

    *max_k_padded_out = ((max_k + 2) / 3) * 3;
    *max_m_out = max_m;
}

static int bt_fpga_prepare_raw_buf(int max_m) {
    bt_fpga_raw_buf = (int32_t *)malloc((size_t)max_m * sizeof(int32_t));
    if (!bt_fpga_raw_buf) {
        fprintf(stderr, "[FPGA] OOM allocating result scratch\n");
        return -1;
    }
    return 0;
}

static int bt_fpga_prepare_scratch_cma(int max_k_padded, int max_m) {
    if (bt_fpga_alloc_cma_buf(&bt_fpga_act_buf,
                              bt_fpga_align_up_size((size_t)max_k_padded, 4096),
                              "act") != 0)
        return -1;
    if (bt_fpga_alloc_cma_buf(&bt_fpga_res_buf,
                              bt_fpga_align_up_size((size_t)max_m * sizeof(int32_t), 4096),
                              "res") != 0)
        return -1;
    return 0;
}

static int bt_fpga_prepare_scratch_devmem(size_t weight_window_bytes,
                                          int max_k_padded,
                                          int max_m) {
    uint32_t act_off;
    uint32_t res_off;
    uint32_t total_needed;

    act_off = (uint32_t)bt_fpga_align_up_size(weight_window_bytes, 4096);
    res_off = act_off +
              (uint32_t)bt_fpga_align_up_size((size_t)max_k_padded, 4096);
    total_needed = res_off +
                   (uint32_t)bt_fpga_align_up_size((size_t)max_m * sizeof(int32_t), 4096);

    if (total_needed > bt_fpga_ddr3_span) {
        fprintf(stderr, "[FPGA] WARNING: need %u bytes but DDR3 span is %u\n",
                total_needed, bt_fpga_ddr3_span);
    }

    bt_fpga_assign_devmem_buf(&bt_fpga_act_buf, act_off, (size_t)max_k_padded);
    bt_fpga_assign_devmem_buf(&bt_fpga_res_buf, res_off, (size_t)max_m * sizeof(int32_t));
    return 0;
}

static uint32_t bt_fpga_advance_weight_cursor(uint32_t cursor,
                                              size_t nib_size,
                                              size_t sign_size) {
    cursor = (uint32_t)bt_fpga_align_up_size(cursor, BT_FPGA_BEAT_BYTES);
    cursor += (uint32_t)nib_size;
    cursor = (uint32_t)bt_fpga_align_up_size(cursor, BT_FPGA_BEAT_BYTES);
    cursor += (uint32_t)sign_size;
    return cursor;
}

static uint32_t bt_fpga_devmem_total_needed(size_t weight_region_bytes,
                                            int max_k_padded,
                                            int max_m) {
    uint32_t act_off;
    uint32_t res_off;

    act_off = (uint32_t)bt_fpga_align_up_size(weight_region_bytes, 4096);
    res_off = act_off +
              (uint32_t)bt_fpga_align_up_size((size_t)max_k_padded, 4096);
    return res_off +
           (uint32_t)bt_fpga_align_up_size((size_t)max_m * sizeof(int32_t), 4096);
}

static int bt_fpga_select_layout(size_t streaming_weight_bytes,
                                 size_t persistent_weight_bytes,
                                 int max_k_padded,
                                 int max_m,
                                 size_t *weight_region_bytes,
                                 uint32_t *total_needed) {
    uint32_t streaming_needed;
    uint32_t persistent_needed;

    if (bt_fpga_mem_backend == BT_FPGA_MEM_CMA) {
        if (bt_fpga_layout_pref == BT_FPGA_LAYOUT_PREF_STREAMING) {
            fprintf(stderr,
                    "[FPGA] BT_FPGA_DDR_LAYOUT=streaming is not supported with CMA; using persistent\n");
        }
        bt_fpga_layout = BT_FPGA_LAYOUT_PERSISTENT;
        if (weight_region_bytes)
            *weight_region_bytes = persistent_weight_bytes;
        if (total_needed)
            *total_needed = 0;
        return 0;
    }

    streaming_needed = bt_fpga_devmem_total_needed(streaming_weight_bytes,
                                                   max_k_padded, max_m);
    persistent_needed = bt_fpga_devmem_total_needed(persistent_weight_bytes,
                                                    max_k_padded, max_m);

    switch (bt_fpga_layout_pref) {
    case BT_FPGA_LAYOUT_PREF_STREAMING:
        bt_fpga_layout = BT_FPGA_LAYOUT_STREAMING;
        if (weight_region_bytes)
            *weight_region_bytes = streaming_weight_bytes;
        if (total_needed)
            *total_needed = streaming_needed;
        return 0;
    case BT_FPGA_LAYOUT_PREF_PERSISTENT:
        if (persistent_needed > bt_fpga_ddr3_span) {
            fprintf(stderr,
                    "[FPGA] BT_FPGA_DDR_LAYOUT=persistent needs %u bytes but mapped span is %u\n",
                    persistent_needed, bt_fpga_ddr3_span);
            return -1;
        }
        bt_fpga_layout = BT_FPGA_LAYOUT_PERSISTENT;
        if (weight_region_bytes)
            *weight_region_bytes = persistent_weight_bytes;
        if (total_needed)
            *total_needed = persistent_needed;
        return 0;
    case BT_FPGA_LAYOUT_PREF_AUTO:
    default:
        if (persistent_needed <= bt_fpga_ddr3_span) {
            bt_fpga_layout = BT_FPGA_LAYOUT_PERSISTENT;
            if (weight_region_bytes)
                *weight_region_bytes = persistent_weight_bytes;
            if (total_needed)
                *total_needed = persistent_needed;
        } else {
            bt_fpga_layout = BT_FPGA_LAYOUT_STREAMING;
            if (weight_region_bytes)
                *weight_region_bytes = streaming_weight_bytes;
            if (total_needed)
                *total_needed = streaming_needed;
            fprintf(stderr,
                    "[FPGA] devmem span %u bytes is too small for persistent layout (%u bytes); falling back to streaming\n",
                    bt_fpga_ddr3_span, persistent_needed);
        }
        return 0;
    }
}

static int bt_fpga_alloc_weight_loc_cma(bt_fpga_wt_loc_t *loc,
                                        size_t nib_size,
                                        size_t sign_size,
                                        int nib_stride,
                                        int sign_stride,
                                        int layer,
                                        const char *weight_name) {
    char name[BITURBO_CMA_NAME_LEN];

    memset(loc, 0, sizeof(*loc));
    loc->nib_stride = nib_stride;
    loc->sign_stride = sign_stride;

    bt_fpga_make_buf_name(name, sizeof(name), layer, weight_name, "nib");
    if (bt_fpga_alloc_cma_buf(&loc->nib, nib_size, name) != 0)
        return -1;

    bt_fpga_make_buf_name(name, sizeof(name), layer, weight_name, "sgn");
    if (bt_fpga_alloc_cma_buf(&loc->sign, sign_size, name) != 0)
        return -1;

    bt_fpga_weight_bytes += loc->nib.size + loc->sign.size;
    return 0;
}

static void bt_fpga_assign_weight_loc_devmem(bt_fpga_wt_loc_t *loc,
                                             uint32_t *cursor,
                                             size_t nib_size,
                                             size_t sign_size,
                                             int nib_stride,
                                             int sign_stride) {
    uint32_t off = *cursor;

    memset(loc, 0, sizeof(*loc));
    loc->nib_stride = nib_stride;
    loc->sign_stride = sign_stride;

    off = (uint32_t)bt_fpga_align_up_size(off, BT_FPGA_BEAT_BYTES);
    bt_fpga_assign_devmem_buf(&loc->nib, off, nib_size);
    off += (uint32_t)nib_size;

    off = (uint32_t)bt_fpga_align_up_size(off, BT_FPGA_BEAT_BYTES);
    bt_fpga_assign_devmem_buf(&loc->sign, off, sign_size);
    off += (uint32_t)sign_size;

    *cursor = off;
}

/* ================================================================
 * Weight preload helpers
 * ================================================================ */

static void bt_fpga_repack_weight_to_loc(const bt_tmac_weight_t *tw,
                                         const bt_fpga_wt_loc_t *loc) {
    int rows = tw->rows;
    int n3_cpu = tw->n3;
    int n3_total = (tw->cols + 2) / 3;
    uint8_t *nib_base = (uint8_t *)loc->nib.cpu;
    uint8_t *sign_base = (uint8_t *)loc->sign.cpu;
    int nib_stride = loc->nib_stride;
    int sign_stride = loc->sign_stride;
    int r;

    memset(nib_base, 0, loc->nib.size);
    memset(sign_base, 0, loc->sign.size);

    for (r = 0; r < rows; r++) {
        const uint8_t *nib_row = tw->three_nib + (size_t)r * tw->nib3_stride;
        const uint8_t *sign_row = tw->three_sign + (size_t)r * tw->sign_stride;
        uint8_t *nib_out = nib_base + (size_t)r * (size_t)nib_stride;
        uint8_t *sign_out = sign_base + (size_t)r * (size_t)sign_stride;
        int tiles_per_row = nib_stride / BT_FPGA_BEAT_BYTES;
        int t;

        for (t = 0; t < tiles_per_row; t++) {
            uint8_t *beat = nib_out + (size_t)t * BT_FPGA_BEAT_BYTES;
            int e;

            for (e = 0; e < BT_FPGA_NUM_ENGINES; e++) {
                int g = t * BT_FPGA_NUM_ENGINES + e;
                int nibble = 0;
                int sign_bit = 0;

                if (g >= n3_total)
                    break;

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

                if (e & 1)
                    beat[e / 2] |= (uint8_t)(nibble << 4);
                else
                    beat[e / 2] |= (uint8_t)(nibble & 0x0F);

                if (sign_bit) {
                    int tile = g / BT_FPGA_NUM_ENGINES;
                    int eng = g % BT_FPGA_NUM_ENGINES;
                    int sign_beat = tile / 4;
                    int chunk = tile % 4;
                    int byte_off = sign_beat * BT_FPGA_BEAT_BYTES + chunk * 4 + eng / 8;
                    sign_out[byte_off] |= (uint8_t)(1 << (eng & 7));
                }
            }
        }
    }
}

static int bt_fpga_copy_btpk_weight(const btpk_weight_t *w,
                                    const bt_fpga_wt_loc_t *loc) {
    if ((size_t)w->nib_size > loc->nib.size || (size_t)w->sign_size > loc->sign.size) {
        fprintf(stderr, "[FPGA] btpk weight buffer smaller than source blob\n");
        return -1;
    }

    memcpy((void *)loc->nib.cpu, w->nib_data, (size_t)w->nib_size);
    memcpy((void *)loc->sign.cpu, w->sign_data, (size_t)w->sign_size);
    return 0;
}

/* ================================================================
 * Layout preparation
 * ================================================================ */

static int bt_fpga_prepare_layout(bt_model_t *model) {
    int n_layers = model->config.n_layers;
    int max_k_padded;
    int max_m;
    size_t max_layer_size = 0;
    size_t total_weight_size = 0;
    size_t weight_region_bytes = 0;
    uint32_t total_cursor = 0;
    int l;

    bt_fpga_reset_layout_state();
    bt_fpga_n_layers = n_layers;
    bt_fpga_layer_locs = (bt_fpga_layer_loc_t *)calloc((size_t)n_layers,
                           sizeof(bt_fpga_layer_loc_t));
    if (!bt_fpga_layer_locs) {
        fprintf(stderr, "[FPGA] OOM allocating layer layout table\n");
        return -1;
    }

    bt_fpga_model_limits(model, &max_k_padded, &max_m);

    for (l = 0; l < n_layers; l++) {
        bt_layer_weights_t *lw = &model->weights.layers[l];
        bt_tmac_weight_t *wts[] = {
            lw->wq.tmac, lw->wk.tmac, lw->wv.tmac, lw->wo.tmac,
            lw->w_gate.tmac, lw->w_up.tmac, lw->w_down.tmac
        };
        uint32_t layer_cursor = 0;
        int i;

        for (i = 0; i < BT_FPGA_WEIGHTS_PER_LAYER; i++) {
            size_t nib_size, sign_size;
            int nib_stride, sign_stride;

            bt_fpga_weight_sizes(wts[i], &nib_size, &sign_size,
                                 &nib_stride, &sign_stride);
            layer_cursor = bt_fpga_advance_weight_cursor(layer_cursor,
                                                         nib_size, sign_size);
            total_cursor = bt_fpga_advance_weight_cursor(total_cursor,
                                                         nib_size, sign_size);
        }

        if ((size_t)layer_cursor > max_layer_size)
            max_layer_size = layer_cursor;
    }
    total_weight_size = (size_t)total_cursor;

    if (bt_fpga_select_layout(max_layer_size, total_weight_size,
                              max_k_padded, max_m,
                              &weight_region_bytes, NULL) != 0)
        goto fail;

    {
        uint32_t persistent_cursor = 0;

        for (l = 0; l < n_layers; l++) {
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
            const char *names[] = { "wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down" };
            uint32_t cursor = 0;
            int i;

            for (i = 0; i < BT_FPGA_WEIGHTS_PER_LAYER; i++) {
                size_t nib_size, sign_size;
                int nib_stride, sign_stride;

                bt_fpga_weight_sizes(wts[i], &nib_size, &sign_size,
                                     &nib_stride, &sign_stride);

                if (bt_fpga_mem_backend == BT_FPGA_MEM_CMA) {
                    if (bt_fpga_alloc_weight_loc_cma(locs[i], nib_size, sign_size,
                                                     nib_stride, sign_stride,
                                                     l, names[i]) != 0)
                        goto fail;
                    bt_fpga_repack_weight_to_loc(wts[i], locs[i]);
                } else if (bt_fpga_layout == BT_FPGA_LAYOUT_PERSISTENT) {
                    bt_fpga_assign_weight_loc_devmem(locs[i], &persistent_cursor,
                                                     nib_size, sign_size,
                                                     nib_stride, sign_stride);
                    bt_fpga_repack_weight_to_loc(wts[i], locs[i]);
                } else {
                    bt_fpga_assign_weight_loc_devmem(locs[i], &cursor,
                                                     nib_size, sign_size,
                                                     nib_stride, sign_stride);
                }
            }
        }
        if (bt_fpga_mem_backend != BT_FPGA_MEM_CMA &&
            bt_fpga_layout == BT_FPGA_LAYOUT_PERSISTENT)
            weight_region_bytes = (size_t)persistent_cursor;
    }

    if (bt_fpga_mem_backend == BT_FPGA_MEM_CMA) {
        if (bt_fpga_prepare_scratch_cma(max_k_padded, max_m) != 0)
            goto fail;
    } else {
        bt_fpga_weight_bytes = weight_region_bytes;
        if (bt_fpga_prepare_scratch_devmem(weight_region_bytes, max_k_padded, max_m) != 0)
            goto fail;
    }

    if (bt_fpga_prepare_raw_buf(max_m) != 0)
        goto fail;

    fprintf(stderr,
            "[FPGA] layout (%s, %s, GGUF): weights=%zu, act=%zu, res=%zu\n",
            bt_fpga_backend_name(bt_fpga_mem_backend),
            bt_fpga_layout_name(bt_fpga_layout),
            bt_fpga_weight_bytes, bt_fpga_act_buf.size, bt_fpga_res_buf.size);
    if (bt_fpga_layout == BT_FPGA_LAYOUT_PERSISTENT) {
        fprintf(stderr,
                "[FPGA] preloaded GGUF weights once: %zu bytes across %d layers\n",
                bt_fpga_weight_bytes, n_layers);
    }
    return 0;

fail:
    bt_fpga_reset_layout_state();
    return -1;
}

static int bt_fpga_prepare_layout_btpk(bt_model_t *model) {
    int n_layers = model->config.n_layers;
    int max_k_padded;
    int max_m;
    size_t max_layer_size = 0;
    size_t total_weight_size = 0;
    size_t weight_region_bytes = 0;
    uint32_t total_cursor = 0;
    int l;

    bt_fpga_reset_layout_state();
    bt_fpga_n_layers = n_layers;
    bt_fpga_layer_locs = (bt_fpga_layer_loc_t *)calloc((size_t)n_layers,
                           sizeof(bt_fpga_layer_loc_t));
    if (!bt_fpga_layer_locs) {
        fprintf(stderr, "[FPGA] OOM allocating layer layout table\n");
        return -1;
    }

    bt_fpga_model_limits(model, &max_k_padded, &max_m);

    for (l = 0; l < n_layers; l++) {
        btpk_layer_t *bl = &model->btpk->layers[l];
        btpk_weight_t *wts[] = {
            &bl->wq, &bl->wk, &bl->wv, &bl->wo,
            &bl->w_gate, &bl->w_up, &bl->w_down
        };
        uint32_t layer_cursor = 0;
        int i;

        for (i = 0; i < BT_FPGA_WEIGHTS_PER_LAYER; i++) {
            size_t nib_size = (size_t)wts[i]->nib_size;
            size_t sign_size = (size_t)wts[i]->sign_size;
            layer_cursor = bt_fpga_advance_weight_cursor(layer_cursor,
                                                         nib_size, sign_size);
            total_cursor = bt_fpga_advance_weight_cursor(total_cursor,
                                                         nib_size, sign_size);
        }

        if ((size_t)layer_cursor > max_layer_size)
            max_layer_size = layer_cursor;
    }
    total_weight_size = (size_t)total_cursor;

    if (bt_fpga_select_layout(max_layer_size, total_weight_size,
                              max_k_padded, max_m,
                              &weight_region_bytes, NULL) != 0)
        goto fail;

    {
        uint32_t persistent_cursor = 0;

        for (l = 0; l < n_layers; l++) {
            btpk_layer_t *bl = &model->btpk->layers[l];
            btpk_weight_t *wts[] = {
                &bl->wq, &bl->wk, &bl->wv, &bl->wo,
                &bl->w_gate, &bl->w_up, &bl->w_down
            };
            bt_fpga_wt_loc_t *locs[] = {
                &bt_fpga_layer_locs[l].wq, &bt_fpga_layer_locs[l].wk,
                &bt_fpga_layer_locs[l].wv, &bt_fpga_layer_locs[l].wo,
                &bt_fpga_layer_locs[l].w_gate, &bt_fpga_layer_locs[l].w_up,
                &bt_fpga_layer_locs[l].w_down
            };
            const char *names[] = { "wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down" };
            uint32_t cursor = 0;
            int i;

            for (i = 0; i < BT_FPGA_WEIGHTS_PER_LAYER; i++) {
                size_t nib_size = (size_t)wts[i]->nib_size;
                size_t sign_size = (size_t)wts[i]->sign_size;

                if (bt_fpga_mem_backend == BT_FPGA_MEM_CMA) {
                    if (bt_fpga_alloc_weight_loc_cma(locs[i], nib_size, sign_size,
                                                     wts[i]->nib_stride, wts[i]->sign_stride,
                                                     l, names[i]) != 0)
                        goto fail;
                    if (bt_fpga_copy_btpk_weight(wts[i], locs[i]) != 0)
                        goto fail;
                } else if (bt_fpga_layout == BT_FPGA_LAYOUT_PERSISTENT) {
                    bt_fpga_assign_weight_loc_devmem(locs[i], &persistent_cursor,
                                                     nib_size, sign_size,
                                                     wts[i]->nib_stride, wts[i]->sign_stride);
                    if (bt_fpga_copy_btpk_weight(wts[i], locs[i]) != 0)
                        goto fail;
                } else {
                    bt_fpga_assign_weight_loc_devmem(locs[i], &cursor,
                                                     nib_size, sign_size,
                                                     wts[i]->nib_stride, wts[i]->sign_stride);
                }
            }
        }
        if (bt_fpga_mem_backend != BT_FPGA_MEM_CMA &&
            bt_fpga_layout == BT_FPGA_LAYOUT_PERSISTENT)
            weight_region_bytes = (size_t)persistent_cursor;
    }

    if (bt_fpga_mem_backend == BT_FPGA_MEM_CMA) {
        if (bt_fpga_prepare_scratch_cma(max_k_padded, max_m) != 0)
            goto fail;
    } else {
        bt_fpga_weight_bytes = weight_region_bytes;
        if (bt_fpga_prepare_scratch_devmem(weight_region_bytes, max_k_padded, max_m) != 0)
            goto fail;
    }

    if (bt_fpga_prepare_raw_buf(max_m) != 0)
        goto fail;

    fprintf(stderr,
            "[FPGA] layout (%s, %s, btpk): weights=%zu, act=%zu, res=%zu\n",
            bt_fpga_backend_name(bt_fpga_mem_backend),
            bt_fpga_layout_name(bt_fpga_layout),
            bt_fpga_weight_bytes, bt_fpga_act_buf.size, bt_fpga_res_buf.size);
    if (bt_fpga_layout == BT_FPGA_LAYOUT_PERSISTENT) {
        fprintf(stderr,
                "[FPGA] preloaded btpk weights once: %zu bytes across %d layers\n",
                bt_fpga_weight_bytes, n_layers);
    }
    return 0;

fail:
    bt_fpga_reset_layout_state();
    return -1;
}

/* ================================================================
 * Runtime helpers
 * ================================================================ */

static void bt_fpga_load_layer(bt_model_t *model, int layer) {
    if (bt_fpga_layout == BT_FPGA_LAYOUT_PERSISTENT)
        return;
    if (bt_fpga_cached_layer == layer)
        return;

    if (model->btpk) {
        btpk_layer_t *bl = &model->btpk->layers[layer];
        btpk_weight_t *wts[] = {
            &bl->wq, &bl->wk, &bl->wv, &bl->wo,
            &bl->w_gate, &bl->w_up, &bl->w_down
        };
        bt_fpga_wt_loc_t *locs[] = {
            &bt_fpga_layer_locs[layer].wq, &bt_fpga_layer_locs[layer].wk,
            &bt_fpga_layer_locs[layer].wv, &bt_fpga_layer_locs[layer].wo,
            &bt_fpga_layer_locs[layer].w_gate, &bt_fpga_layer_locs[layer].w_up,
            &bt_fpga_layer_locs[layer].w_down
        };
        int i;

        for (i = 0; i < BT_FPGA_WEIGHTS_PER_LAYER; i++) {
            memcpy((void *)locs[i]->nib.cpu, wts[i]->nib_data, (size_t)wts[i]->nib_size);
            memcpy((void *)locs[i]->sign.cpu, wts[i]->sign_data, (size_t)wts[i]->sign_size);
        }
    } else {
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
        int i;

        for (i = 0; i < BT_FPGA_WEIGHTS_PER_LAYER; i++)
            bt_fpga_repack_weight_to_loc(wts[i], locs[i]);
    }

    bt_fpga_cached_layer = layer;
}

static void bt_fpga_upload_activations(const int8_t *q8_buf, int K) {
    int k_padded = ((K + 2) / 3) * 3;
    uint8_t *dst = (uint8_t *)bt_fpga_act_buf.cpu;

    memcpy(dst, q8_buf, (size_t)K);
    memset(dst + K, 0, (size_t)(k_padded - K));
}

static void bt_fpga_gemv(const bt_tmac_weight_t *tw,
                         const bt_fpga_wt_loc_t *loc,
                         int32_t *results) {
    int M = tw->rows;
    int K = ((tw->cols + 2) / 3) * 3;
    uint32_t act_phys = bt_fpga_act_buf.dma_addr;
    uint32_t res_phys = bt_fpga_res_buf.dma_addr;
    int rows_done = 0;

    while (rows_done < M) {
        int tile_m = M - rows_done;
        uint32_t tile_nib;
        uint32_t tile_sign;

        if (tile_m > BT_FPGA_MAX_DIM_M)
            tile_m = BT_FPGA_MAX_DIM_M;

        tile_nib = loc->nib.dma_addr + (uint32_t)rows_done * (uint32_t)loc->nib_stride;
        tile_sign = loc->sign.dma_addr + (uint32_t)rows_done * (uint32_t)loc->sign_stride;

        bt_fpga_reg_write(BT_REG_NIB_BASE, tile_nib);
        bt_fpga_reg_write(BT_REG_SIGN_BASE, tile_sign);
        bt_fpga_reg_write(BT_REG_DIM_M, (uint32_t)tile_m);
        bt_fpga_reg_write(BT_REG_DIM_K, (uint32_t)K);
        bt_fpga_reg_write(BT_REG_DIM_N3, (uint32_t)(K / 3));
        bt_fpga_reg_write(BT_REG_ACT_DDR3_BASE, act_phys);
        bt_fpga_reg_write(BT_REG_RES_DDR3_BASE, res_phys);
        bt_fpga_reg_write(BT_REG_CTRL, BT_CTRL_START | BT_CTRL_DDR3_MODE);

        if (bt_fpga_wait_done() < 0) {
            fprintf(stderr, "[FPGA] GEMV timeout M=%d K=%d tile_offset=%d\n",
                    M, K, rows_done);
            memset(&results[rows_done], 0, (size_t)tile_m * sizeof(int32_t));
            rows_done += tile_m;
            continue;
        }

        memcpy(&results[rows_done], (const void *)bt_fpga_res_buf.cpu,
               (size_t)tile_m * sizeof(int32_t));
        rows_done += tile_m;
    }
}

static void bt_fpga_gemv_dequant(float *out, const bt_tmac_weight_t *tw,
                                 const bt_fpga_wt_loc_t *loc,
                                 float inv_scale) {
    int r;
    float dequant;

    bt_fpga_gemv(tw, loc, bt_fpga_raw_buf);
    dequant = inv_scale * tw->scale;
    for (r = 0; r < tw->rows; r++)
        out[r] = (float)bt_fpga_raw_buf[r] * dequant;
}

static void bt_fpga_gemv_dequant_btpk(float *out, const btpk_weight_t *w,
                                      const bt_fpga_wt_loc_t *loc,
                                      float inv_scale) {
    int M = w->rows;
    int K = w->k_padded;
    uint32_t act_phys = bt_fpga_act_buf.dma_addr;
    uint32_t res_phys = bt_fpga_res_buf.dma_addr;
    int rows_done = 0;
    float dequant;
    int r;

    while (rows_done < M) {
        int tile_m = M - rows_done;
        uint32_t tile_nib;
        uint32_t tile_sign;

        if (tile_m > BT_FPGA_MAX_DIM_M)
            tile_m = BT_FPGA_MAX_DIM_M;

        tile_nib = loc->nib.dma_addr + (uint32_t)rows_done * (uint32_t)loc->nib_stride;
        tile_sign = loc->sign.dma_addr + (uint32_t)rows_done * (uint32_t)loc->sign_stride;

        bt_fpga_reg_write(BT_REG_NIB_BASE, tile_nib);
        bt_fpga_reg_write(BT_REG_SIGN_BASE, tile_sign);
        bt_fpga_reg_write(BT_REG_DIM_M, (uint32_t)tile_m);
        bt_fpga_reg_write(BT_REG_DIM_K, (uint32_t)K);
        bt_fpga_reg_write(BT_REG_DIM_N3, (uint32_t)(K / 3));
        bt_fpga_reg_write(BT_REG_ACT_DDR3_BASE, act_phys);
        bt_fpga_reg_write(BT_REG_RES_DDR3_BASE, res_phys);
        bt_fpga_reg_write(BT_REG_CTRL, BT_CTRL_START | BT_CTRL_DDR3_MODE);

        if (bt_fpga_wait_done() < 0) {
            fprintf(stderr, "[FPGA] btpk GEMV timeout M=%d K=%d off=%d\n",
                    M, K, rows_done);
            memset(&bt_fpga_raw_buf[rows_done], 0,
                   (size_t)tile_m * sizeof(int32_t));
            rows_done += tile_m;
            continue;
        }

        memcpy(&bt_fpga_raw_buf[rows_done], (const void *)bt_fpga_res_buf.cpu,
               (size_t)tile_m * sizeof(int32_t));
        rows_done += tile_m;
    }

    dequant = inv_scale * w->scale;
    for (r = 0; r < M; r++)
        out[r] = (float)bt_fpga_raw_buf[r] * dequant;
}

#endif /* BITURBO_FPGA_H */
