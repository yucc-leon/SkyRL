"""
Long-context attention benchmark: SDPA vs npu_fusion_attention vs triton-ascend FA.
Uses the official triton-ascend fused attention kernel from:
https://ascend.github.io/triton-ascend/sources/getting-started/tutorials/04-fused-attention.html

Usage:
    source /usr/local/Ascend/cann-8.5.0/bin/setenv.bash
    ASCEND_RT_VISIBLE_DEVICES=7 python npu_support/bench_attn_v2.py
"""
import time, torch, torch_npu, triton, triton.language as tl

DEVICE = "npu"
DTYPE = torch.bfloat16
WARMUP, REPEATS = 3, 10

# ─── Official triton-ascend flash attention kernel (forward only) ───
@triton.jit
def _attn_fwd_inner(acc_ptr, l_i, m_i, q, K_block_ptr, V_block_ptr,
                    start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, tl.trans(k))
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk = qk * qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        p = tl.math.exp(qk)
        p_cast = p.to(k.dtype)
        v = tl.load(V_block_ptr)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc_ptr = acc_ptr * alpha[:, None]
        acc_ptr = tl.dot(p_cast, v, acc_ptr)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc_ptr, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, M, Out, acc, sm_scale,
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,
              Z: tl.constexpr, H: tl.constexpr, N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr):
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H
    pid = tl.program_id(0)
    for block_idx in range(pid, NUM_BLOCKS, 20):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        Q_block_ptr = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk), offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
        V_block_ptr = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk), offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
        K_block_ptr = tl.make_block_ptr(base=K + qvk_offset, shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk), offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
        O_block_ptr = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on), offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc_ptr = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        q = tl.load(Q_block_ptr)
        if STAGE & 1:
            acc_ptr, l_i, m_i = _attn_fwd_inner(acc_ptr, l_i, m_i, q, K_block_ptr, V_block_ptr,
                task_m_idx, sm_scale, BLOCK_M, HEAD_DIM, BLOCK_N,
                4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5)
        if STAGE & 2:
            acc_ptr, l_i, m_i = _attn_fwd_inner(acc_ptr, l_i, m_i, q, K_block_ptr, V_block_ptr,
                task_m_idx, sm_scale, BLOCK_M, HEAD_DIM, BLOCK_N,
                2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.
float8e5)
        m_i += tl.math.log(l_i)
        accumulator = acc_ptr / l_i[:, None]
        m_ptrs = M + task_hz_idx * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))


def triton_attention(q, k, v, causal, sm_scale, BM=128, BN=128):
    """Wrapper for the triton-ascend flash attention kernel."""
    o = torch.empty_like(q)
    acc = torch.zeros_like(q, dtype=torch.float32)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    stage = 3 if causal else 1
    _attn_fwd[(20,)](
        q, k, v, M, o, acc, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], N_CTX=q.shape[2], HEAD_DIM=q.shape[3],
        BLOCK_M=BM, BLOCK_N=BN, STAGE=stage)
    return o


# ─── Benchmark harness ───
def bench(fn, label, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        try: fn()
        except Exception as e:
            print(f"  {label:50s}  FAILED: {type(e).__name__}: {e}")
            return None
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / repeats * 1000
    print(f"  {label:50s}  {ms:10.2f} ms")
    return ms


def main():
    torch.npu.set_device(0)
    D = 128
    sm_scale = 1.0 / (D ** 0.5)

    configs = [
        # (B, H, S)
        (1, 32, 2048),
        (1, 32, 8192),
        (1, 32, 16384),
        (1, 32, 32768),
        (1, 16, 32768),
        (1, 32, 65536),
    ]

    print("=" * 85)
    print("Long-Context Attention Benchmark: SDPA vs npu_fusion vs triton-ascend FA")
    print(f"  dtype={DTYPE}, head_dim={D}, causal=True, warmup={WARMUP}, repeats={REPEATS}")
    print("=" * 85)

    results = []
    for B, H, S in configs:
        print(f"\n--- B={B}, H={H}, S={S}, D={D} ---")
        q = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
        k = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
        v = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)

        # SDPA
        t1 = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True), "SDPA")

        # npu_fusion_attention (BSH layout)
        q_bsh = q.transpose(1, 2).contiguous().view(B, S, H * D)
        k_bsh = k.transpose(1, 2).contiguous().view(B, S, H * D)
        v_bsh = v.transpose(1, 2).contiguous().view(B, S, H * D)
        t2 = bench(lambda: torch_npu.npu_fusion_attention(
            q_bsh, k_bsh, v_bsh, head_num=H, input_layout="BSH",
            scale=sm_scale, pre_tockens=65536, next_tockens=0), "npu_fusion_attention (BSH)")

        # npu_fusion_attention (BNSD layout, no reshape needed)
        t2b = bench(lambda: torch_npu.npu_fusion_attention(
            q, k, v, H, input_layout="BNSD",
            scale=sm_scale, pre_tockens=65536, next_tockens=0), "npu_fusion_attention (BNSD)")

        # triton-ascend FA — HEAD_DIM=128 needs smaller blocks to fit in UB
        # HEAD_DIM=128: BM=32,BN=64 verified working
        # HEAD_DIM=64:  BM=64,BN=128 (official tutorial)
        BM = 32 if D >= 128 else 64
        BN = 64 if D >= 128 else 128
        t3 = bench(lambda: triton_attention(q, k, v, causal=True, sm_scale=sm_scale, BM=BM, BN=BN),
                   "triton-ascend FA")

        results.append((S, H, t1, t2, t2b, t3))

    # Summary
    print(f"\n{'=' * 85}")
    print(f"{'S':>8} {'H':>4}  {'SDPA':>10}  {'npu_fus BSH':>12}  {'npu_fus BNSD':>13}  {'triton FA':>10}  {'winner':>10}")
    print("-" * 85)
    for S, H, t1, t2, t2b, t3 in results:
        vals = {}
        if t1: vals["SDPA"] = t1
        if t2: vals["npu_BSH"] = t2
        if t2b: vals["npu_BNSD"] = t2b
        if t3: vals["triton"] = t3
        winner = min(vals, key=vals.get) if vals else "N/A"
        print(f"{S:>8} {H:>4}  {t1 or 0:>10.2f}  {t2 or 0:>12.2f}  {t2b or 0:>13.2f}  {t3 or 0:>10.2f}  {winner:>10}")
    print("=" * 85)


if __name__ == "__main__":
    main()
