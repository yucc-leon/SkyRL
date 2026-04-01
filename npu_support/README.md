# SkyRL NPU (Ascend) Support

This directory contains everything needed to run SkyRL on Huawei Ascend NPUs.

## How it works

SkyRL is written for CUDA. Instead of forking and patching every file, we use a
single monkey-patch module (`patch_cuda.py`) that replaces `torch.cuda` with a
thin proxy to `torch.npu` **before any other code runs**. This means:

- **Zero changes to SkyRL source code**
- **Zero changes to downstream projects** (e.g. codescout)
- The patch is applied once at process startup via `import npu_support.patch_cuda`

## Quick Start

```bash
# 1. Set up the conda environment (one-time)
bash npu_support/setup_env.sh

# 2. In your training script, add this as the VERY FIRST import:
#    import npu_support.patch_cuda
#    (see codescout/scripts/run_async_training_npu.sh for a complete example)
```

## What gets patched

| Original (CUDA)                    | Replacement (NPU)                  |
|------------------------------------|-------------------------------------|
| `torch.cuda.*`                     | `torch.npu.*` (via module proxy)    |
| `DeviceMesh("cuda", ...)`          | `DeviceMesh("npu", ...)`            |
| `flash_attn.bert_padding.*`        | no-op stubs (flash_attn unavailable)|
| `CUDA_VISIBLE_DEVICES`             | `ASCEND_RT_VISIBLE_DEVICES`         |
| NCCL backend                       | HCCL backend                        |

## Requirements

- CANN 8.5.0+
- torch 2.8.0 + torch_npu 2.8.0
- vllm 0.11.0 + vllm-ascend 0.11.0rc1
- Python 3.12
