"""
Benchmark: SDPA vs npu_fusion_attention vs triton-ascend flash attention on Ascend NPU.

Usage:
    source /usr/local/Ascend/cann-8.5.0/bin/setenv.bash
    python npu_support/bench_attention.py
"""
import time
import torch
import torch_npu

DEVICE = torch.device("npu:0")
DTYPE = torch.bfloat16
WARMUP = 5
REPEATS = 20


def bench(fn, label, warmup=WARMUP, repeats=REPEATS):
    """Benchmark a function, return avg ms."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / repeats * 1000
    print(f"  {label:45s}  {elapsed_ms:8.2f} ms")
    return elapsed_ms


def run_sdpa(q, k, v, causal=True):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal
    )


def run_npu_fusion(q_bsh, k_bsh, v_bsh, head_num, head_dim, causal=True):
    return torch_npu.npu_fusion_attention(
        q_bsh, k_bsh, v_bsh,
        head_num=head_num,
        input_layout="BSH",
        scale=1.0 / (head_dim ** 0.5),
        pre_tockens=65536,
        next_tockens=0 if causal else 65536,
    )


def main():
    torch.npu.set_device(0)

    configs = [
        # (batch, heads, seq_len, head_dim)
        (1,  32, 512,   128),
        (1,  32, 2048,  128),
        (1,  32, 4096,  128),
        (2,  32, 2048,  128),
        (1,  20, 2048,  128),   # Qwen3-4B: 36 heads, 4 KV heads → 20 Q heads per GQA group? Just test 20
    ]

    # Check triton-ascend availability
    has_triton = False
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
        has_triton = True
    except ImportError:
        pass

    print("=" * 72)
    print("Attention Benchmark on Ascend NPU")
    print(f"  dtype: {DTYPE}, warmup: {WARMUP}, repeats: {REPEATS}")
    print(f"  triton-ascend: {'available' if has_triton else 'NOT available'}")
    print("=" * 72)

    for B, H, S, D in configs:
        print(f"\n--- B={B}, H={H}, S={S}, D={D} (total {B*H*S*D*2/1e6:.1f} MB per QKV) ---")

        # SDPA format: (B, H, S, D)
        q = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
        k = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
        v = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)

        # npu_fusion_attention format: (B, S, H*D) = BSH
        q_bsh = q.transpose(1, 2).contiguous().view(B, S, H * D)
        k_bsh = k.transpose(1, 2).contiguous().view(B, S, H * D)
        v_bsh = v.transpose(1, 2).contiguous().view(B, S, H * D)

        # 1. SDPA
        bench(lambda: run_sdpa(q, k, v), "PyTorch SDPA (is_causal=True)")

        # 2. npu_fusion_attention
        bench(lambda: run_npu_fusion(q_bsh, k_bsh, v_bsh, H, D), "npu_fusion_attention (BSH, causal)")

        # 3. SDPA without causal (for reference)
        bench(lambda: run_sdpa(q, k, v, causal=False), "PyTorch SDPA (no causal)")

        # Correctness check: compare SDPA vs npu_fusion_attention
        out_sdpa = run_sdpa(q, k, v)
        out_npu = run_npu_fusion(q_bsh, k_bsh, v_bsh, H, D)[0]
        out_npu_bhsd = out_npu.view(B, S, H, D).transpose(1, 2)
        max_diff = (out_sdpa.float() - out_npu_bhsd.float()).abs().max().item()
        print(f"  {'max |SDPA - npu_fusion|':45s}  {max_diff:.6f}")

    # Backward pass benchmark (training-relevant)
    print(f"\n{'=' * 72}")
    print("Backward pass benchmark (B=1, H=32, S=2048, D=128)")
    print("=" * 72)
    B, H, S, D = 1, 32, 2048, 128

    q = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    k = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    v = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def sdpa_fwd_bwd():
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.sum().backward()

    bench(sdpa_fwd_bwd, "SDPA fwd+bwd")

    q_bsh = torch.randn(B, S, H * D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    k_bsh = torch.randn(B, S, H * D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    v_bsh = torch.randn(B, S, H * D, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def npu_fusion_fwd_bwd():
        out = torch_npu.npu_fusion_attention(
            q_bsh, k_bsh, v_bsh,
            head_num=H, input_layout="BSH",
            scale=1.0 / (D ** 0.5),
            pre_tockens=65536, next_tockens=0,
        )
        out[0].sum().backward()

    bench(npu_fusion_fwd_bwd, "npu_fusion_attention fwd+bwd")

    print(f"\n{'=' * 72}")
    print("Done!")


if __name__ == "__main__":
    main()
