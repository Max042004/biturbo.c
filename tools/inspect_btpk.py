"""Quick dumper for the btpk_header_t layout — confirms the new LM head
SVD offsets/sizes land where the C struct expects them."""

import struct
import sys
from pathlib import Path

# Must mirror biturbo_btpk.h exactly (little-endian, naturally aligned).
HEADER_FMT = (
    "<"
    "8s"    # magic[8]
    "I"     # version
    "I"     # format
    "I"     # num_engines
    "I"     # beat_bytes
    "i"     # dim
    "i"     # n_layers
    "i"     # n_heads
    "i"     # n_kv_heads
    "i"     # vocab_size
    "i"     # ffn_dim
    "i"     # max_seq_len
    "f"     # norm_eps
    "f"     # rope_theta
    "i"     # tok_vocab_size
    "i"     # tok_max_token_len
    "i"     # tok_bos_id
    "i"     # tok_eos_id
    "i"     # tok_eot_id
    "i"     # _tok_pad
    "4x"    # C struct pads before the next uint64_t (6 x int32 ends at +4 mod 8)
    "Q"     # tokenizer_off
    "Q"     # tokenizer_size
    "Q"     # token_embed_off
    "Q"     # token_embed_size
    "Q"     # final_norm_off
    "Q"     # final_norm_size
    "i"     # lm_head_rank
    "i"     # _lm_head_pad
    "Q"     # lm_head_V_off
    "Q"     # lm_head_V_size
    "Q"     # lm_head_E_q_off
    "Q"     # lm_head_E_q_size
    "Q"     # lm_head_E_scale_off
    "Q"     # lm_head_E_scale_size
    "Q"     # layers_off
    "Q"     # total_file_size
)

FIELDS = [
    "magic", "version", "format", "num_engines", "beat_bytes",
    "dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size",
    "ffn_dim", "max_seq_len", "norm_eps", "rope_theta",
    "tok_vocab_size", "tok_max_token_len", "tok_bos_id",
    "tok_eos_id", "tok_eot_id", "_tok_pad",
    "tokenizer_off", "tokenizer_size",
    "token_embed_off", "token_embed_size",
    "final_norm_off", "final_norm_size",
    "lm_head_rank", "_lm_head_pad",
    "lm_head_V_off", "lm_head_V_size",
    "lm_head_E_q_off", "lm_head_E_q_size",
    "lm_head_E_scale_off", "lm_head_E_scale_size",
    "layers_off", "total_file_size",
]


def main():
    if len(sys.argv) != 2:
        print("Usage: inspect_btpk.py <file.btpk>")
        return 1
    p = Path(sys.argv[1])
    data = p.read_bytes()
    print(f"file: {p}  ({len(data) / 1e6:.2f} MB)")
    print(f"sizeof(header) per struct fmt: {struct.calcsize(HEADER_FMT)} bytes")
    values = struct.unpack_from(HEADER_FMT, data, 0)
    for name, val in zip(FIELDS, values):
        if isinstance(val, bytes):
            val = val.rstrip(b"\x00").decode("ascii", "replace")
        print(f"  {name:>24} = {val}")

    # Cross-check that the three SVD sections are in the expected byte ranges
    r = dict(zip(FIELDS, values))
    if r["lm_head_rank"] > 0:
        D, V, rank = r["dim"], r["vocab_size"], r["lm_head_rank"]
        want_V = D * rank * 4
        want_Eq = V * rank
        want_Es = V * 4
        ok = (r["lm_head_V_size"] == want_V
              and r["lm_head_E_q_size"] == want_Eq
              and r["lm_head_E_scale_size"] == want_Es)
        print(f"\nexpected V={want_V} E_q={want_Eq} E_scale={want_Es}: {'OK' if ok else 'MISMATCH'}")
        # Spot-check first 16 bytes of each section
        for name in ("lm_head_V", "lm_head_E_q", "lm_head_E_scale"):
            off = r[f"{name}_off"]
            size = r[f"{name}_size"]
            head = data[off:off + 16].hex()
            print(f"  {name} @ 0x{off:x} [{size} B], first 16B: {head}")
    else:
        print("\nlm_head_rank=0 — no SVD sidecars in this .btpk")

    return 0


if __name__ == "__main__":
    sys.exit(main())
