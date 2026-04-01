"""
Long-context attention benchmark on Ascend NPU.
Tests SDPA, npu_fusion_attention, and triton-ascend flash attention
at sequence lengths relevant to agentic RL training (8k-64k).

Usage:
    source /usr/local/Ascend/cann-8.5.0/bin/setenv.bash
    ASCEND_RT_VISIBLE_DEVICES=7 python npu_support/bench_attention_long.py

Uses NPU 7 by default to avoid interfering with training on 0-6.
"""
import os
import time
import torch
import torch_npu

DEVICE = torch.device("npu:0")
DTYPE = torch.bfloat16
WARMUP = 3
REPEATS = 10

# Qwen3-4B config: 36 heads, 4 KV heads, head_dim=128
# For GQA, Q has 36 heads, KV has 4 heads. We benchmark the Q side.
QWEN3_4B_HEADS = 36
QWEN3_4B_KV_HEADS = 4
QWEN3_4B_HEAD_DIM = 128


def bench(fn, label, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        try:
            fn()
        except Exception as e:
            print(f"  {label:50s}  FAILED: {e}")
            return None
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / repeats * 1000
    print(f"  {label:50s}  {ms:10.2f} ms")
    return ms


def bench_sdpa(B, H, S, D, causal=True):
    q = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
    return bench(
        lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal),
        f"SDPA (B={B},H={H},S={S},D={D})"
    )


def bench_npu_fusion(B, H, S, D, causal=True):
    q = torch.randn(B, S, H * D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B, S, H * D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B, S, H * D, dtype=DTYPE, device=DEVICE)
    return bench(
        lambda: torch_npu.npu_fusion_attention(
            q, k, v, head_num=H, input_layout="BSH",
            scale=1.0 / (D ** 0.5),
            pre_tockens=65536, next_tockens=0 if causal else 65536,
        ),
        f"npu_fusion_attn (B={B},H={H},S={S},D={D})"
    )


def bench_triton_fa(B, H, S, D, causal=True):
    """Benchmark triton-ascend flash attention if available."""
    try:
        import triton
        import triton.language as tl
    except ImportError:
        print(f"  {'triton FA (not installed)':50s}  SKIPPED")
        return None

    # Use the official triton-ascend flash attention tutorial kernel
    # We import it dynamically from the ascend examples
    try:
        # Try importing the pre-built flash attention from triton-ascend
        from triton.runtime import driver
        
        # Build a minimal flash attention kernel inline
        # Based on https://ascend.github.io/triton-ascend/sources/getting-started/tutorials/04-fused-attention.html
        @triton.jit
        def _fwd_kernel(
            Q, K, V, Out,
            stride_qb, stride_qh, stride_qm, stride_qk,
            stride_kb, stride_kh, stride_kn, stride_kk,
            stride_vb, stride_vh, stride_vn, stride_vk,
            stride_ob, stride_oh, stride_om, stride_ok,
            N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
            IS_CAUSAL: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_bh = tl.program_id(1)
            off_b = pid_bh // tl.num_programs(1)
            off_h = pid_bh % tl.num_programs(1)

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, HEAD_DIM)

            q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
            q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX)

            m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

            num_blocks_n = (N_CTX + BLOCK_N - 1) // BLOCK_N
            for j in range(num_blocks_n):
                start_n = j * BLOCK_N
                offs_n_j = start_n + tl.arange(0, BLOCK_N)

                k_ptrs = K + off_b * stride_kb + off_h * stride_kh + offs_n_j[:, None] * stride_kn + offs_k[None, :] * stride_kk
                k = tl.load(k_ptrs, mask=offs_n_j[:, None] < N_CTX)

                qk = tl.dot(q, tl.trans(k))
                qk *= 1.0 / tl.sqrt(HEAD_DIM * 1.0)

                if IS_CAUSAL:
                    mask = offs_m[:, None] >= offs_n_j[None, :]
                    qk = tl.where(mask, qk, float("-inf"))

                m_ij = tl.max(qk, axis=1)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(qk - m_new[:, None])

                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None]

                v_ptrs = V + off_b * stride_vb + off_h * stride_vh + offs_n_j[:, None] * stride_vn + offs_k[None, :] * stride_vk
                v = tl.load(v_ptrs, mask=offs_n_j[:, None] < N_CTX)
                acc += tl.dot(p.to(v.dtype), v)

                m_i = m_new

            acc = acc / l_i[:, None]
            out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
            tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)

        q = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
        k = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
        v = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
        out = torch.empty_like(q)

        BLOCK_M = min(128, S)
        BLOCK_N = min(128, S)
        grid = (S // BLOCK_M, B * H)

        def run():
            _fwd_kernel[grid](
                q, k, v, out,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                N_CTX=S, HEAD_DIM=D,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                IS_CAUSAL=causal,
            )

        return bench(run, f"triton FA (B={B},H={H},S={S},D={D})")

    except Exception as e:
        print(f"  {'triton FA':50s}  ERROR: {e}")
        return None


def bench_sdpa_fwd_bwd(B, H, S, D):
    q = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    k = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    v = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    def run():
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.sum().backward()
        q.grad = k.grad = v.grad = None
    return bench(run, f"SDPA fwd+bwd (B={B},H={H},S={S},D={D})")


def bench_npu_fusion_fwd_bwd(B, H, S, D):
    q = torch.randn(B, S, H * D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    k = torch.randn(B, S, H * D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    v = torch.randn(B, S, H * D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    def run():
        out = torch_npu.npu_fusion_attention(
            q, k, v, head_num=H, input_layout="BSH",
            scale=1.0 / (D ** 0.5),
            pre_tockens=65536, next_tockens=0,
        )
        out[0].sum().backward()
        q.grad = k.grad = v.grad = None
    return bench(run, f"npu_fusion fwd+bwd (B={B},H={H},S={S},D={D})")


def main():
    torch.npu.set_device(0)

    print("=" * 80)
    print("Long-Context Attention Benchmark on Ascend NPU")
    print(f"  dtype: {DTYPE}, warmup: {WARMUP}, repeats: {REPEATS}")
    print(f"  Qwen3-4B config: {QWEN3_4B_HEADS}h, {QWEN3_4B_HEAD_DIM}d")

    has_triton = False
    try:
        import triton
        has_triton = True
        print(f"  triton-ascend: {triton.__version__}")
    except ImportError:
        print(f"  triton-ascend: NOT available")
    print("=" * 80)

    # ---- Forward-only benchmark ----
    # Use Qwen3-4B-like config but with fewer heads to fit in memory at long seqs
    configs = [
        # (B, H, S, D) — realistic agentic training lengths
        (1, 32, 2048,  128),   # baseline
        (1, 32, 8192,  128),   # 8k
        (1, 32, 16384, 128),   # 16k
        (1, 32, 32768, 128),   # 32k — typical agentic
        (1, 16, 32768, 128),   # 32k fewer heads (GQA KV side)
        (1, 32, 65536, 128),   # 64k — long agentic
    ]

    print(f"\n{'─' * 80}")
    print("Forward pass (causal)")
    print(f"{'─' * 80}")

    results = []
    for B, H, S, D in configs:
        mem_mb = B * H * S * D * 2 * 3 / 1e6  # 3 tensors, bf16
        print(f"\n  B={B}, H={H}, S={S}, D={D}  (QKV: {mem_mb:.0f} MB)")

        t_sdpa = bench_sdpa(B, H, S, D)
        t_npu = bench_npu_fusion(B, H, S, D)
        t_triton = bench_triton_fa(B, H, S, D) if has_triton else None

        results.append((B, H, S, D, t_sdpa, t_npu, t_triton))

    # ---- Forward+Backward benchmark (training-relevant) ----
    print(f"\n{'─' * 80}")
    print("Forward + Backward (causal) — training-relevant")
    print(f"{'─' * 80}")

    bwd_configs = [
        (1, 32, 2048,  128),
        (1, 32, 8192,  128),
        (1, 32, 16384, 128),
        (1, 32, 32768, 128),
    ]

    for B, H, S, D in bwd_configs:
        print(f"\n  B={B}, H={H}, S={S}, D={D}")
        bench_sdpa_fwd_bwd(B, H, S, D)
        bench_npu_fusion_fwd_bwd(B, H, S, D)

    # ---- Summary ----
    print(f"\n{'=' * 80}")
    print("Summary (forward only, ms)")
    print(f"{'S':>8s}  {'SDPA':>10s}  {'npu_fusion':>10s}  {'triton_FA':>10s}  {'winner':>12s}")
    print(f"{'─' * 60}")
    for B, H, S, D, t_sdpa, t_npu, t_triton in results:
        vals = {"SDPA": t_sdpa, "npu_fusion": t_npu, "triton_FA": t_triton}
        valid = {k: v for k, v in vals.items() if v is not None}
        winner = min(valid, key=valid.get) if valid else "N/A"
        print(f"{S:>8d}  {t_sdpa or 0:>10.2f}  {t_npu or 0:>10.2f}  {t_triton or 0:>10.2f}  {winner:>12s}")
    print("=" * 80)


if __name__ == "__main__":
    main()
