# biturbo

Zero-dependency BitNet 1.58-bit inference engine in C with TurboQuant KV cache.

## What it does

Runs Microsoft's [BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) model directly from GGUF files. The full inference path is implemented from scratch in portable C99:

| Stage | Component | Implementation |
|-------|-----------|----------------|
| 1 | Tokenization | GPT-2 BPE with byte-to-unicode mapping |
| 2 | Token embedding | F16 mmap'd from GGUF |
| 3 | RMS norm | Pre-attention normalization |
| 4 | BitLinear input quantization | Per-token dynamic INT8 |
| 5 | Ternary weights | I2_S group-interleaved {-1, 0, +1} |
| 6 | BitLinear GEMM | INT8 x ternary accumulation |
| 7 | Q/K/V projection | Separate matrices |
| 8 | RoPE | Rotary position embedding (theta=500k) |
| 9 | KV cache | TurboQuant 4-bit (RHT + Lloyd-Max + QJL) |
| 10 | Attention | GQA (20 query / 5 KV heads) |
| 11 | Sub-norm + output | RMS norm + BitLinear projection |
| 12 | FFN gate | SqReLU-gated GLU |
| 13 | FFN down | Sub-norm + BitLinear projection |
| 14 | LM head | Tied to token embedding (F16) |
| 15 | Sampling | Temperature + top-p nucleus |

## TurboQuant KV cache

The KV cache uses the full [TurboQuant](https://arxiv.org/abs/2504.19874) pipeline instead of naive uniform quantization.

Quantization path for each K/V vector:

1. L2 normalize the vector.
2. Apply a Random Hadamard Transform (RHT) to decorrelate channels.
3. Quantize with a Lloyd-Max 3-bit codebook.
4. Store a 1-bit QJL sign hash for the residual.

Attention-time reconstruction:

- K attention uses a two-stage inner product estimate in rotated space.
- V reconstruction uses MSE-oriented pointwise dequantization.
- Storage is 72 bytes per 128-element block, or 4.5 bits per element.

## Build

Host build:

```bash
make
make debug
```

This requires only a C99 compiler. No extra runtime dependencies are needed.

## Download model

The engine loads GGUF files with I2_S (1.58-bit ternary) weights.

```bash
pip install huggingface-hub

huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
    --include "ggml-model-i2_s.gguf" \
    --local-dir model/
```

Or convert from Microsoft's BF16 checkpoint:

```bash
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir model-bf16/
python BitNet/utils/convert-ms-to-gguf-bitnet.py model-bf16/ --outtype i2_s
```

## Pre-pack GGUF for FPGA (.btpk)

`pack_btpk` converts a GGUF model into a standalone `.btpk` file whose ternary weight blobs are already striped for the DE10-Nano T-MAC accelerator.

Build the packer:

```bash
make
```

Convert GGUF to `.btpk`:

```bash
./pack_btpk model/ggml-model-i2_s.gguf model/ggml-model.btpk
```

The `.btpk` format stores tokenizer data, token embeddings, norms, and pre-striped FPGA weight blobs so the board does not need to repack weights at runtime.

## Run

CPU path:

```bash
./biturbo model/ggml-model-i2_s.gguf -p "Where is Tokyo?" -n 64
./biturbo model/ggml-model-i2_s.gguf -p "Explain quantum computing" -n 256 -t 0.0
```

FPGA path:

```bash
make fpga
./biturbo_fpga model/ggml-model.btpk -p "hi" -n 6
```

The CPU-only `./biturbo` executable expects GGUF and will reject `.btpk`.

## DE10-Nano FPGA path

The FPGA build now supports two memory backends and two DDR3 layout modes:

- backend `devmem`: reserved DDR carveout mapped through `/dev/mem`
- backend `cma`: Linux CMA-backed DMA allocation through `/dev/biturbo-cma`
- layout `streaming`: one small weight window reused on each layer switch
- layout `persistent`: every weight gets a stable DDR address and is loaded once

Persistent inference is no longer tied to CMA. The `cma` backend is always persistent, and the `devmem` backend can also run persistent when the reserved DDR span is large enough for all weights plus activation/result scratch.

### FPGA userspace build

```bash
make fpga
```

This builds `biturbo_fpga` for the Cortex-A9 on DE10-Nano with `BT_FPGA` enabled.

### Runtime backend selection

`biturbo_fpga` checks `BT_FPGA_MEM_BACKEND`:

- `auto` or unset: try CMA first, then fall back to legacy `devmem`
- `cma`: require `/dev/biturbo-cma`
- `devmem`: force the reserved-memory `/dev/mem` backend

`biturbo_fpga` also checks `BT_FPGA_DDR_LAYOUT`:

- `auto` or unset: choose `persistent` when the mapped DDR span can hold the full model weights plus scratch, otherwise fall back to `streaming`
- `persistent`: require enough reserved DDR for the whole weight set
- `streaming`: force the old layer-window behavior even on a large carveout

On a board with a dedicated reserved carveout, `BT_FPGA_MEM_BACKEND=devmem BT_FPGA_DDR_LAYOUT=persistent` avoids Linux CMA entirely while keeping weights resident across tokens.

### Optional CMA driver build

Build on target, or build against a matching DE10-Nano kernel tree:

```bash
make fpga
make cma-module KDIR=/lib/modules/$(uname -r)/build
sudo insmod kernel/biturbo_cma.ko
ls -l /dev/biturbo-cma
```

For Windows + WSL cross-builds:

```powershell
wsl bash -lc "cd /mnt/c/intelFPGA_lite/18.1/ghrd_bitnet/biturbo.c && make cma-module KDIR=~/src/linux-socfpga-4.14.73-ltsi ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf-"
```

The repository `Makefile` injects the ARMv7 module flags needed for older 4.14 ARM builds with modern hard-float GCC toolchains.

### Device tree

For the persistent `devmem` path, reserve a `no-map` carveout large enough for BitNet weights plus scratch buffers. The default region in this repo is:

```text
base = 0x24000000
span = 0x1C000000
```

Relevant node shape:

```dts
reserved-memory {
    #address-cells = <1>;
    #size-cells = <1>;
    ranges;

    biturbo_fpga_reserved: biturbo-fpga@24000000 {
        reg = <0x24000000 0x1c000000>;
        compatible = "shared-dma-pool";
        no-map;
    };
};
```

If you still want the optional CMA driver, bind that same reserved region through a platform node:

```dts
biturbo_cma {
    compatible = "biturbo,cma-pool";
    memory-region = <&biturbo_fpga_reserved>;
    dma-coherent;
    status = "okay";
};
```

After updating the DTB that the board actually boots, verify on target:

```bash
ls /proc/device-tree/reserved-memory/
hexdump -Cv /proc/device-tree/reserved-memory/biturbo-fpga@24000000/reg
grep -i -A3 -B1 24000000 /proc/iomem
```

### Example run

```bash
BT_FPGA_MEM_BACKEND=devmem BT_FPGA_DDR_LAYOUT=persistent sudo ./biturbo_fpga model/ggml-model.btpk -p "hi" -n 6
```

Expected persistent-weight log pattern:

```text
[FPGA] T-MAC accelerator bound: backend=devmem, CPU DDR3 0x24000000, AVM base 0x24000000, span 0x1C000000
[FPGA] layout (devmem, persistent, btpk): weights=<weight_bytes>, act=<act_bytes>, res=<res_bytes>
[FPGA] preloaded btpk weights once: 440647680 bytes across 30 layers
```

If you still see a 32 MB DDR span warning or `layout (..., streaming, ...)`, the board is still using the old small carveout path.

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `-p` | `"Hello"` | Input prompt |
| `-n` | 256 | Max tokens to generate |
| `-t` | 0.8 | Temperature (`0.0` = greedy) |
| `-k` | 0.9 | Top-p nucleus sampling |
| `-s` | time | RNG seed |

## Architecture

```text
biturbo.h             Types, config, API
biturbo.c             Full inference engine and profiling output
biturbo_fpga.h        FPGA backend with CMA/devmem support
biturbo_cma_ioctl.h   Shared userspace/kernel ioctl ABI
kernel/biturbo_cma.c  CMA misc driver
main.c                CLI runner
Makefile              Userspace + kernel-module build entry points
```

## Model specs

BitNet-b1.58-2B-4T:

| Parameter | Value |
|-----------|-------|
| Hidden dim | 2560 |
| Layers | 30 |
| Attention heads | 20 query / 5 KV |
| Head dim | 128 |
| FFN dim | 6912 |
| Vocab size | 128256 |
| Context length | 4096 |
| Weight format | 1.58-bit ternary (I2_S) |
| Parameters | about 2.4B |

## Performance

Host reference, single-threaded on Apple M1:

| Metric | Value |
|--------|-------|
| Speed | about 1.3 tok/s |
| Model memory | about 1.1 GB mmap'd |
| KV cache | about 80 MB |
| Runtime buffers | about 15 MB |

### DE10-Nano FPGA profiling

Measured with `.btpk`, prompt `hi`, generating 5 tokens:

| Configuration | Total | Transformer layers | LM head | Sampling |
|---------------|-------|--------------------|---------|----------|
| Legacy streaming layer window | 78.33 s | 30.02 s (6.00 s/token) | 47.68 s (9.54 s/token) | 0.63 s (0.1262 s/token) |
| Persistent weights in DDR | 56.38 s | 8.08 s (1.62 s/token) | 47.67 s (9.53 s/token) | 0.63 s (0.1260 s/token) |

The persistent weight path cuts transformer layer time by about 3.7x. After that improvement, the dominant bottleneck on DE10-Nano becomes the CPU-side LM head, which is still around 9.5 seconds per generated token.

The built-in profile summary prints:

```text
biturbo: profile (generated tokens): layers=8.08s (1.62 s/token), lm_head=47.67s (9.53 s/token), sampling=0.63s (0.1260 s/token)
```

## References

- [BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) - Microsoft's official 1.58-bit model
- [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453) - BitNet architecture paper
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) - BitNet b1.58 paper
- [TurboQuant](https://arxiv.org/abs/2504.19874) - KV cache quantization with RHT + QJL
