#!/usr/bin/env python3
"""
compute_lmhead_svd.py — offline tool to pre-compute a rank-r SVD factoring of
the Q6_K `token_embd.weight` tensor in a biturbo GGUF, for use as a tied LM
head approximation.

Output sidecars (raw, little-endian):
  <out>/lm_head_V.f32       — [dim, r] float32, row-major
  <out>/lm_head_E_q.i8      — [vocab, r] int8, row-major
  <out>/lm_head_E_scale.f32 — [vocab] float32 (per-row E scale)
  <out>/lm_head_meta.json   — {"dim": D, "vocab": V, "rank": r, "source": "..."}

These files are ingested by `pack_btpk --lmhead-svd-dir <out>`.

Math:
  E ≈ U_r Σ_r V_rᵀ          (truncated SVD of the Q6_K-dequantized embedding)
  logits = E h ≈ (U_r Σ_r) · (V_rᵀ h)
         = E_prod · h_proj                (runtime: two small GEMVs)

We compute the SVD via eigendecomposition of the small D×D Gram matrix C = EᵀE
(D=2560), which gives V_r as its top-r eigenvectors. Then E_prod = E V_r.
This avoids touching the full V×V Gram and is numerically stable for PSD C.

Usage:
  python tools/compute_lmhead_svd.py <gguf_path> --rank 512 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

# --- GGUF minimal reader ----------------------------------------------------

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_TYPE_U8, GGUF_TYPE_I8, GGUF_TYPE_U16, GGUF_TYPE_I16 = 0, 1, 2, 3
GGUF_TYPE_U32, GGUF_TYPE_I32, GGUF_TYPE_F32, GGUF_TYPE_BOOL = 4, 5, 6, 7
GGUF_TYPE_STR, GGUF_TYPE_ARR = 8, 9
GGUF_TYPE_U64, GGUF_TYPE_I64, GGUF_TYPE_F64 = 10, 11, 12

_TYPE_FIXED_SIZE = {
    GGUF_TYPE_U8: 1, GGUF_TYPE_I8: 1, GGUF_TYPE_BOOL: 1,
    GGUF_TYPE_U16: 2, GGUF_TYPE_I16: 2,
    GGUF_TYPE_U32: 4, GGUF_TYPE_I32: 4, GGUF_TYPE_F32: 4,
    GGUF_TYPE_U64: 8, GGUF_TYPE_I64: 8, GGUF_TYPE_F64: 8,
}

GGUF_TENSOR_Q6K = 14


class Reader:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: memoryview, pos: int = 0):
        self.buf = buf
        self.pos = pos

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def u64(self) -> int:
        v = struct.unpack_from("<Q", self.buf, self.pos)[0]
        self.pos += 8
        return v

    def f32(self) -> float:
        v = struct.unpack_from("<f", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def f64(self) -> float:
        v = struct.unpack_from("<d", self.buf, self.pos)[0]
        self.pos += 8
        return v

    def string(self) -> str:
        n = self.u64()
        s = bytes(self.buf[self.pos:self.pos + n]).decode("utf-8", "replace")
        self.pos += n
        return s

    def skip_value(self, vtype: int) -> None:
        if vtype == GGUF_TYPE_STR:
            self.string()
        elif vtype == GGUF_TYPE_ARR:
            etype = self.u32()
            n = self.u64()
            if etype == GGUF_TYPE_STR:
                for _ in range(n):
                    self.string()
            else:
                self.pos += n * _TYPE_FIXED_SIZE[etype]
        else:
            self.pos += _TYPE_FIXED_SIZE[vtype]


def parse_gguf(path: Path):
    """Return (mmap_bytes, tensors_dict, data_base_offset).

    tensors_dict maps name -> (dtype, dims_tuple, offset_in_data_section).
    """
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_BINARY", 0))
    size = os.fstat(fd).st_size
    import mmap
    mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
    os.close(fd)

    r = Reader(memoryview(mm))
    magic = r.u32()
    if magic != GGUF_MAGIC:
        raise ValueError(f"not a GGUF file: magic=0x{magic:08x}")
    version = r.u32()
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    n_tensors = r.u64()
    n_kv = r.u64()

    # Skip all KV metadata — we only need tensor headers for the embedding
    for _ in range(n_kv):
        _ = r.string()                # key
        vtype = r.u32()
        r.skip_value(vtype)

    tensors = {}
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.u32()
        dims = tuple(r.u64() for _ in range(n_dims))
        dtype = r.u32()
        offset = r.u64()
        tensors[name] = (dtype, dims, offset)

    # Data section is 32-byte aligned after tensor headers (matches biturbo.c:2496)
    alignment = 32
    data_base = r.pos
    pad = (alignment - data_base % alignment) % alignment
    data_base += pad

    return mm, tensors, data_base


# --- Q6_K dequantizer -------------------------------------------------------

# Block layout (matches biturbo.c:214-219):
#   uint8_t  ql[128]       # lower 4 bits (2 quants per byte)
#   uint8_t  qh[64]        # upper 2 bits (4 quants per byte)
#   int8_t   scales[16]    # per-16-element sub-block scale (signed)
#   float16  d             # super-block scale
# => 210 bytes per 256-element block.

Q6K_BLOCK_BYTES = 128 + 64 + 16 + 2  # 210


def dequant_q6k_rows(blob: bytes, n_rows: int, dim: int,
                     chunk_rows: int = 4000) -> np.ndarray:
    """Dequantize a contiguous Q6_K blob to float32 [n_rows, dim].

    Streams through the data in row-chunks to keep peak scratch memory bounded;
    the output buffer itself is n_rows*dim*4 bytes, which the caller must fit.
    """
    assert dim % 256 == 0, "Q6_K requires dim divisible by 256"
    nb_per_row = dim // 256
    row_bytes = nb_per_row * Q6K_BLOCK_BYTES
    assert len(blob) >= n_rows * row_bytes, \
        f"blob too small: {len(blob)} < {n_rows * row_bytes}"

    out = np.empty((n_rows, dim), dtype=np.float32)

    # Per-l scale index: for l in 0..31, is_idx = l // 16 (0 or 1)
    is_idx = np.arange(32) // 16  # (32,)

    # Dequant one chunk of rows at a time
    for r0 in range(0, n_rows, chunk_rows):
        r1 = min(r0 + chunk_rows, n_rows)
        nc = r1 - r0
        # Extract the chunk as raw uint8 blocks: (nc*nb_per_row, 210)
        chunk_start = r0 * row_bytes
        chunk_end = r1 * row_bytes
        raw = np.frombuffer(blob, dtype=np.uint8,
                            count=(chunk_end - chunk_start),
                            offset=chunk_start)
        nb_chunk = nc * nb_per_row
        raw = raw.reshape(nb_chunk, Q6K_BLOCK_BYTES)

        ql = raw[:, 0:128]                      # (nb_chunk, 128) uint8
        qh = raw[:, 128:192]                    # (nb_chunk, 64)  uint8
        scales = raw[:, 192:208].view(np.int8)  # (nb_chunk, 16)  int8
        # f16 super-block scale: last 2 bytes as uint16 → view as float16
        d_u16 = np.ascontiguousarray(raw[:, 208:210]).view(np.uint16).reshape(nb_chunk)
        d = d_u16.view(np.float16).astype(np.float32)  # (nb_chunk,)

        # Output buffer for this chunk: (nb_chunk, 256) float32
        y = np.empty((nb_chunk, 256), dtype=np.float32)

        for sub in range(2):  # n = 0 (first 128 elts) and n = 128 (second 128)
            off_ql = sub * 64          # 0 or 64
            off_qh = sub * 32          # 0 or 32
            off_sc = sub * 8           # 0 or 8
            y_off = sub * 128          # 0 or 128

            ql_l0  = ql[:, off_ql:off_ql + 32]       # l in [0,32), position l
            ql_l32 = ql[:, off_ql + 32:off_ql + 64]  #                position l+32
            qh_l   = qh[:, off_qh:off_qh + 32]       # shared

            # Low 4 bits of ql combined with 2 bits of qh → 6 bits in [0,63].
            # Subtract 32 → signed [-32, 31].
            q1 = ((ql_l0 & 0x0F) | ((qh_l >> 0) & 0x03) << 4).astype(np.int32) - 32
            q2 = ((ql_l32 & 0x0F) | ((qh_l >> 2) & 0x03) << 4).astype(np.int32) - 32
            q3 = ((ql_l0 >> 4)   | ((qh_l >> 4) & 0x03) << 4).astype(np.int32) - 32
            q4 = ((ql_l32 >> 4)  | ((qh_l >> 6) & 0x03) << 4).astype(np.int32) - 32

            # Scale lookup per l: sc[:, is_idx + {0,2,4,6}]
            sc_q1 = scales[:, off_sc + is_idx + 0].astype(np.int32)
            sc_q2 = scales[:, off_sc + is_idx + 2].astype(np.int32)
            sc_q3 = scales[:, off_sc + is_idx + 4].astype(np.int32)
            sc_q4 = scales[:, off_sc + is_idx + 6].astype(np.int32)

            dcol = d[:, None]  # (nb_chunk, 1)

            y[:, y_off +  0:y_off + 32]  = dcol * (sc_q1 * q1).astype(np.float32)
            y[:, y_off + 32:y_off + 64]  = dcol * (sc_q2 * q2).astype(np.float32)
            y[:, y_off + 64:y_off + 96]  = dcol * (sc_q3 * q3).astype(np.float32)
            y[:, y_off + 96:y_off + 128] = dcol * (sc_q4 * q4).astype(np.float32)

        # Reshape back from (nc*nb_per_row, 256) to (nc, dim)
        out[r0:r1] = y.reshape(nc, dim)

    return out


# --- SVD + INT8 quantization ------------------------------------------------

def truncated_svd_via_gram(E: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (V_r [D,r], E_prod [V,r]) with E ≈ E_prod @ V_r.T."""
    V, D = E.shape
    assert rank <= D

    print(f"[svd] gram matrix E^T E (D={D}) ...", flush=True)
    t0 = time.time()
    # Do the Gram in f32 (cheap, ~26 MB for D=2560) then promote only the
    # small D×D result to f64 for the eigen solve. A single f32→f64 cast of
    # the full V×D E would need several GB, which this machine does not have.
    C = (E.T @ E).astype(np.float64, copy=False)
    print(f"[svd] C: {C.shape}, took {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending
    print(f"[svd] eigh done in {time.time() - t0:.1f}s", flush=True)
    # Top-r in descending order
    idx = np.argsort(eigvals)[::-1][:rank]
    eigvals_top = np.clip(eigvals[idx], a_min=0.0, a_max=None)
    V_r = eigvecs[:, idx].astype(np.float32, copy=False)  # (D, r)

    # Energy captured
    total = float(np.sum(np.clip(eigvals, 0.0, None)))
    captured = float(np.sum(eigvals_top))
    print(f"[svd] rank={rank} captures {captured / total * 100:.2f}% "
          f"of spectral energy (sum of sigma^2)", flush=True)

    t0 = time.time()
    # E_prod = E @ V_r = U_r * Σ_r  (since E = U Σ V^T, E V = U Σ).
    E_prod = (E @ V_r).astype(np.float32, copy=False)
    print(f"[svd] E @ V_r in {time.time() - t0:.1f}s, shape={E_prod.shape}",
          flush=True)
    return V_r, E_prod


def quantize_rows_int8(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric INT8 quant: scale[v] = max|row| / 127, q = round(row/scale)."""
    assert mat.dtype == np.float32
    abs_max = np.max(np.abs(mat), axis=1)  # (V,)
    scale = (abs_max / 127.0).astype(np.float32)
    # Guard zero rows (use tiny scale to avoid div-by-zero; the row is all zeros anyway)
    safe = np.where(scale > 0, scale, np.float32(1.0))
    q = np.clip(np.round(mat / safe[:, None]), -127, 127).astype(np.int8)
    # Rows that were all zero keep scale=0 — dequant still recovers zero logits.
    return q, scale


# --- Sanity check -----------------------------------------------------------

def sanity_check(E: np.ndarray, V_r: np.ndarray, E_q: np.ndarray,
                 E_scale: np.ndarray, E_prod: np.ndarray | None = None,
                 n_samples: int = 1000,
                 topk: tuple[int, ...] = (1, 5, 10, 50, 100),
                 batch: int = 64, seed: int = 0) -> dict:
    """Compare exact vs approximate logits on n_samples proxy hidden states.

    Pick random token embedding rows (optionally perturbed) as stand-ins for
    the hidden state `h` that feeds the LM head; this is the most natural
    proxy because the final hidden state after `output_norm` lies in the same
    space as the token embeddings.

    The comparison is batched so the (batch, V) intermediates stay small
    (~30 MB at batch=64 with V=128256).
    """
    V, D = E.shape
    rng = np.random.default_rng(seed)
    idx = rng.choice(V, size=n_samples, replace=False)
    h_all = E[idx].copy()
    noise = rng.standard_normal(size=h_all.shape).astype(np.float32) \
        * (0.1 * float(np.std(h_all)))
    h_all[::2] += noise[::2]

    # Pre-dequantize E_q to f32 once (263 MB for V=128256, r=512) so the
    # approx matmul can use BLAS with a contiguous operand.
    E_approx_int8 = E_q.astype(np.float32) * E_scale[:, None]  # (V, r)

    # Track overlap for two approximations side-by-side:
    #   "int8"  — what runtime actually computes (E_q * scale)
    #   "f32"   — ideal low-rank approx without INT8 loss (E_prod directly)
    hit_counts_int8 = {k: 0 for k in topk}
    hit_counts_f32  = {k: 0 for k in topk}
    argmax_hits_int8 = 0
    argmax_hits_f32  = 0
    # Shortlist recall: does the TRUE top-1 fall inside approx top-K?
    # Tight metric for deciding whether "SVD shortlist + exact rescore" works.
    recall_int8 = {k: 0 for k in topk}
    recall_f32  = {k: 0 for k in topk}

    have_f32 = E_prod is not None

    t0 = time.time()
    for b0 in range(0, n_samples, batch):
        b1 = min(b0 + batch, n_samples)
        h = h_all[b0:b1]

        exact = h @ E.T                             # (b, V)
        h_proj = h @ V_r                            # (b, r)
        approx_int8 = h_proj @ E_approx_int8.T      # (b, V)
        if have_f32:
            approx_f32 = h_proj @ E_prod.T          # (b, V)

        ex_arg = np.argmax(exact, axis=1)
        argmax_hits_int8 += int(np.sum(ex_arg == np.argmax(approx_int8, axis=1)))
        if have_f32:
            argmax_hits_f32 += int(np.sum(ex_arg == np.argmax(approx_f32, axis=1)))

        for k in topk:
            ex_top = np.argpartition(-exact, kth=k - 1, axis=1)[:, :k]
            ap_top_i8 = np.argpartition(-approx_int8, kth=k - 1, axis=1)[:, :k]
            if have_f32:
                ap_top_f32 = np.argpartition(-approx_f32, kth=k - 1, axis=1)[:, :k]
            for i in range(ex_top.shape[0]):
                hit_counts_int8[k] += int(
                    np.intersect1d(ex_top[i], ap_top_i8[i]).size)
                if have_f32:
                    hit_counts_f32[k] += int(
                        np.intersect1d(ex_top[i], ap_top_f32[i]).size)
                # Shortlist recall: is the exact-top-1 inside the approx-top-k?
                if ex_arg[i] in ap_top_i8[i]:
                    recall_int8[k] += 1
                if have_f32 and ex_arg[i] in ap_top_f32[i]:
                    recall_f32[k] += 1
    print(f"[sanity] exact+approx scoring on {n_samples} proxies "
          f"(batch={batch}) took {time.time() - t0:.1f}s", flush=True)

    results = {}
    print(f"[sanity] {'k':>4}   overlap(i8)  overlap(f32)   recall(i8)  recall(f32)",
          flush=True)
    for k in topk:
        rate_i8  = hit_counts_int8[k] / (n_samples * k)
        rate_f32 = hit_counts_f32[k]  / (n_samples * k) if have_f32 else -1.0
        rec_i8   = recall_int8[k] / n_samples
        rec_f32  = recall_f32[k]  / n_samples if have_f32 else -1.0
        results[f"top{k}_overlap_int8"] = rate_i8
        results[f"top{k}_recall_int8"] = rec_i8
        if have_f32:
            results[f"top{k}_overlap_f32"] = rate_f32
            results[f"top{k}_recall_f32"]  = rec_f32
        r_f32 = f"{rec_f32 * 100:>8.2f}%" if have_f32 else "   n/a"
        o_f32 = f"{rate_f32 * 100:>8.2f}%" if have_f32 else "   n/a"
        print(f"[sanity] {k:>4}       {rate_i8 * 100:>6.2f}%     {o_f32}     "
              f"{rec_i8 * 100:>6.2f}%   {r_f32}", flush=True)

    results["top1_argmax_int8"] = argmax_hits_int8 / n_samples
    if have_f32:
        results["top1_argmax_f32"] = argmax_hits_f32 / n_samples
    print(f"[sanity] top-1 argmax agreement: int8={results['top1_argmax_int8'] * 100:.2f}%"
          + (f", f32={results['top1_argmax_f32'] * 100:.2f}%" if have_f32 else ""),
          flush=True)

    return results


# --- main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path, help="path to ggml-model-*.gguf")
    ap.add_argument("--rank", "-r", type=int, default=512)
    ap.add_argument("--out", "-o", type=Path, required=True,
                    help="output sidecar directory (will be created)")
    ap.add_argument("--min-top1", type=float, default=0.95,
                    help="abort if top-1 argmax agreement below this")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--skip-sanity", action="store_true")
    args = ap.parse_args()

    if not args.gguf.is_file():
        print(f"error: {args.gguf} not found", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[load] mapping {args.gguf} ...", flush=True)
    mm, tensors, data_base = parse_gguf(args.gguf)
    print(f"[load] {len(tensors)} tensors, data section @ 0x{data_base:x}",
          flush=True)

    if "token_embd.weight" not in tensors:
        print("error: token_embd.weight not found in GGUF", file=sys.stderr)
        return 2
    dtype, dims, off = tensors["token_embd.weight"]
    if dtype != GGUF_TENSOR_Q6K:
        print(f"error: token_embd.weight is dtype {dtype}, expected Q6_K (14)",
              file=sys.stderr)
        return 2
    # GGUF stores dims in row-major but reversed for matmul convention:
    # token_embd.weight has dims [dim, vocab] in file but semantically [vocab, dim].
    # biturbo.c treats it row-major as [vocab][dim]. Trust file layout:
    #   row_bytes = nb_per_row * Q6K_BLOCK_BYTES; total = vocab * row_bytes.
    # From dims: dims[0] = dim (inner), dims[1] = vocab. Verify from file size.
    # We prefer to derive vocab and dim explicitly from known biturbo.c conventions.
    inner, outer = int(dims[0]), int(dims[1])
    dim, vocab = inner, outer
    print(f"[load] token_embd.weight dims raw={dims} -> vocab={vocab}, dim={dim}",
          flush=True)

    if dim % 256 != 0:
        print(f"error: dim={dim} not divisible by 256", file=sys.stderr)
        return 2
    if args.rank % 16 != 0:
        # Runtime INT8 GEMV loads 16 bytes per NEON iteration; pad rank up.
        print(f"error: --rank {args.rank} must be a multiple of 16",
              file=sys.stderr)
        return 2

    # Byte range for the embedding in the mmap'd file
    nb_per_row = dim // 256
    row_bytes = nb_per_row * Q6K_BLOCK_BYTES
    blob_bytes = vocab * row_bytes
    blob_start = data_base + off
    blob = bytes(memoryview(mm)[blob_start:blob_start + blob_bytes])
    print(f"[load] embedding blob: {blob_bytes / 1e6:.1f} MB "
          f"(offset=0x{blob_start:x})", flush=True)

    t0 = time.time()
    print(f"[dequant] Q6_K → f32 [{vocab}, {dim}] "
          f"(~{vocab * dim * 4 / 1e9:.2f} GB) ...", flush=True)
    E = dequant_q6k_rows(blob, vocab, dim)
    print(f"[dequant] done in {time.time() - t0:.1f}s", flush=True)

    # Quick sanity: E should have non-trivial magnitude per row
    row_norms = np.linalg.norm(E, axis=1)
    print(f"[dequant] row-norm stats: mean={row_norms.mean():.3f} "
          f"min={row_norms.min():.3f} max={row_norms.max():.3f}", flush=True)

    # Truncated SVD
    V_r, E_prod = truncated_svd_via_gram(E, args.rank)

    # INT8 quantize E_prod per-row
    print("[quant] per-row INT8 on E_prod ...", flush=True)
    E_q, E_scale = quantize_rows_int8(E_prod)
    print(f"[quant] E_q dtype={E_q.dtype} shape={E_q.shape}, "
          f"E_scale dtype={E_scale.dtype} shape={E_scale.shape}", flush=True)

    # Optional sanity check — pass E_prod so we can separate quant vs rank loss
    sanity = None
    if not args.skip_sanity:
        sanity = sanity_check(E, V_r, E_q, E_scale, E_prod=E_prod,
                              n_samples=args.samples)
    # E_prod is large (~263 MB at r=512) — drop it before writing sidecars.
    del E_prod
    import gc
    gc.collect()

    # Write sidecars (raw little-endian, row-major)
    def _write_raw(path: Path, arr: np.ndarray) -> None:
        assert arr.flags["C_CONTIGUOUS"]
        arr.tofile(path)
        print(f"[write] {path.name}: {arr.nbytes / 1e6:.2f} MB "
              f"(dtype={arr.dtype}, shape={arr.shape})", flush=True)

    _write_raw(args.out / "lm_head_V.f32",       np.ascontiguousarray(V_r))
    _write_raw(args.out / "lm_head_E_q.i8",      np.ascontiguousarray(E_q))
    _write_raw(args.out / "lm_head_E_scale.f32", np.ascontiguousarray(E_scale))

    meta = {
        "dim": int(dim),
        "vocab": int(vocab),
        "rank": int(args.rank),
        "source": str(args.gguf.resolve()),
        "dtype_V": "float32",
        "dtype_E_q": "int8",
        "dtype_E_scale": "float32",
    }
    if sanity is not None:
        meta["sanity"] = sanity
    (args.out / "lm_head_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[write] {(args.out / 'lm_head_meta.json').name}", flush=True)

    # Enforce the accuracy gate unless user asked to skip it
    if sanity is not None and sanity["top1_argmax_int8"] < args.min_top1:
        print(f"\nFAILED accuracy gate: top-1 argmax(int8) "
              f"{sanity['top1_argmax_int8'] * 100:.2f}% < {args.min_top1 * 100:.1f}%",
              file=sys.stderr)
        print("Try a higher --rank, or set --min-top1 lower to override.",
              file=sys.stderr)
        return 1

    print(f"\nOK. Sidecars written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
