# biturbo

Zero-dependency BitNet 1.58-bit inference engine in C with TurboQuant KV cache.

## What it does

Runs Microsoft's [BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) model directly from GGUF files. All 15 inference stages implemented from scratch in ~1500 lines of portable C99:

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
| 9 | KV cache | **TurboQuant 4-bit** (RHT + Lloyd-Max + QJL) |
| 10 | Attention | GQA (20 query / 5 KV heads) |
| 11 | Sub-norm + output | RMS norm + BitLinear projection |
| 12 | FFN gate | SqReLU-gated GLU |
| 13 | FFN down | Sub-norm + BitLinear projection |
| 14 | LM head | Tied to token embedding (F16) |
| 15 | Sampling | Temperature + top-p nucleus |

## TurboQuant KV cache

The KV cache uses the full [TurboQuant](https://arxiv.org/abs/2504.19874) pipeline instead of naive uniform quantization:

**Quantize** (K and V at each position):
1. L2 normalize the vector
2. Random Hadamard Transform (RHT) to decorrelate channels
3. Lloyd-Max 3-bit codebook quantization (8 optimal centroids for N(0,1))
4. QJL 1-bit sign hash on quantization residual

**K attention** (two-stage inner product estimation):
- RHT(query) and QJL projection pre-computed once per query head
- Stage 1: codebook dot product in rotated space
- Stage 2: QJL XNOR correction: `(2*agree - d) * ||r|| * sqrt(pi/2) / d`
- Final score: `||k|| * (mse_dot + qjl_correction)`

**V dequant** (MSE-only point-wise reconstruction):
- Codebook lookup in rotated space -> inverse RHT -> scale by norm

72 bytes per 128-element block (4.5 bits/element). Provably unbiased inner product estimation.

## Build

```bash
make
```

Requires only a C99 compiler. No dependencies. ARM NEON auto-detected on Apple Silicon / aarch64.

```bash
make debug    # with AddressSanitizer + UBSan
```

## Download model

The engine loads GGUF files with I2_S (1.58-bit ternary) weights. Download the official BitNet model:

```bash
# Install huggingface CLI if needed
pip install huggingface-hub

# Download the GGUF model (~1.1 GB)
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
    --include "ggml-model-i2_s.gguf" \
    --local-dir model/
```

Or convert from the BF16 weights using Microsoft's [BitNet](https://github.com/microsoft/BitNet) conversion tools:

```bash
# Download BF16 weights
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir model-bf16/

# Convert to GGUF I2_S (requires BitNet repo)
python BitNet/utils/convert-ms-to-gguf-bitnet.py model-bf16/ --outtype i2_s
```

## Run

```bash
./biturbo model/ggml-model-i2_s.gguf -p "Where is Tokyo?" -n 64

./biturbo model/ggml-model-i2_s.gguf -p "Once upon a time" -n 128 -t 0.7

./biturbo model/ggml-model-i2_s.gguf -p "Explain quantum computing" -n 256 -t 0.0
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-p` | `"Hello"` | Input prompt |
| `-n` | 256 | Max tokens to generate |
| `-t` | 0.8 | Temperature (0.0 = greedy) |
| `-k` | 0.9 | Top-p nucleus sampling |
| `-s` | time | RNG seed |

## Architecture

```
biturbo.h    Types, config, API (bt_kv_block_t, bt_i2s_weight_t, etc.)
biturbo.c    Full inference engine (~1500 lines)
main.c       CLI runner
Makefile     Build with NEON auto-detection
```

### BitNet-b1.58-2B-4T model specs

| Parameter | Value |
|-----------|-------|
| Hidden dim | 2560 |
| Layers | 30 |
| Attention heads | 20 (query) / 5 (KV) |
| Head dim | 128 |
| FFN dim | 6912 |
| Vocab size | 128,256 |
| Context length | 4,096 |
| Weight format | 1.58-bit ternary (I2_S) |
| Parameters | ~2.4B |
| Model size | ~1.1 GB (GGUF) |

## Performance

Single-threaded on Apple M1:

| Metric | Value |
|--------|-------|
| Speed | ~1.3 tok/s |
| Model memory | ~1.1 GB (mmap'd) |
| KV cache | ~80 MB (30 layers x 5 heads x 4096 positions) |
| Runtime buffers | ~15 MB |

## References

- [BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) — Microsoft's official 1.58-bit model
- [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453) — BitNet architecture paper
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) — BitNet b1.58 paper
- [TurboQuant](https://arxiv.org/abs/2504.19874) — KV cache quantization with RHT + QJL
