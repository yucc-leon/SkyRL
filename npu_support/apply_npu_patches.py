"""Apply NPU device patches to SkyRL source files.
Usage: python apply_npu_patches.py <skyrl_train_dir>
Example: python apply_npu_patches.py SkyRL/skyrl-train/skyrl_train
"""
import sys, os

base = sys.argv[1] if len(sys.argv) > 1 else "skyrl-train/skyrl_train"
patches = {
    "distributed/fsdp_utils.py": [
        ('x = x.to_empty(device=torch.cuda.current_device(), recurse=False)',
         'x = x.to_empty(device=torch.npu.current_device(), recurse=False)'),
        ('torch.cuda.empty_cache()', 'torch.npu.empty_cache()'),
        ('device_id = torch.cuda.current_device()\n',
         'device_id = torch.npu.current_device()\n'),
        ('handle.flat_param_to(torch.device(f"cuda:{device_id}"), non_blocking=True)',
         'handle.flat_param_to(torch.device(f"npu:{device_id}"), non_blocking=True)'),
        ('device = torch.cuda.current_device()\n    model.to(device',
         'device = torch.npu.current_device()\n    model.to(device'),
        ('full_param = full_param.detach().cuda()',
         'full_param = full_param.detach().npu()'),
        ('full_tensor = torch.empty(sharded_param.size(), device="cuda"',
         'full_tensor = torch.empty(sharded_param.size(), device="npu"'),
        ('device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,)',
         'device_mesh = init_device_mesh("npu", mesh_shape=(world_size,)'),
        ('device_mesh = init_device_mesh(\n            "cuda"',
         'device_mesh = init_device_mesh(\n            "npu"'),
    ],
    "distributed/fsdp_strategy.py": [
        ('torch.cuda.manual_seed_all(seed)', 'torch.npu.manual_seed_all(seed)'),
        ('torch.cuda.set_device(local_rank)', 'torch.npu.set_device(local_rank)'),
        ('torch.cuda.synchronize()', 'torch.npu.synchronize()'),
        ('torch.cuda.empty_cache()', 'torch.npu.empty_cache()'),
        ('device_id=torch.cuda.current_device()', 'device_id=torch.npu.current_device()'),
        ('load_fsdp_optimizer(optimizer, torch.cuda.current_device())',
         'load_fsdp_optimizer(optimizer, torch.npu.current_device())'),
    ],
    "distributed/strategy.py": [
        ('data = data.to(torch.cuda.current_device())',
         'data = data.to(torch.npu.current_device())'),
        ('ret = [torch.zeros_like(data).to(torch.cuda.current_device())',
         'ret = [torch.zeros_like(data).to(torch.npu.current_device())'),
        ('dist.all_gather(ret, data.to(torch.cuda.current_device()))',
         'dist.all_gather(ret, data.to(torch.npu.current_device()))'),
        ('if torch.cuda.is_available() and torch.cuda.device_count() > 0:',
         'if torch.npu.is_available() and torch.npu.device_count() > 0:'),
        ('rng_state["cuda"] = torch.cuda.get_rng_state()',
         'rng_state["npu"] = torch.npu.get_rng_state()'),
        ('"cuda" in rng_state and torch.cuda.is_available() and torch.cuda.device_count() > 0:',
         '"npu" in rng_state and torch.npu.is_available() and torch.npu.device_count() > 0:'),
        ('torch.cuda.set_rng_state(rng_state["cuda"])',
         'torch.npu.set_rng_state(rng_state["npu"])'),
    ],
    "workers/fsdp/fsdp_worker.py": [
        ('torch.distributed.get_rank() % torch.cuda.device_count()',
         'torch.distributed.get_rank() % torch.npu.device_count()'),
        ('torch.cuda.empty_cache()', 'torch.npu.empty_cache()'),
        ('device = torch.cuda.current_device()', 'device = torch.npu.current_device()'),
    ],
    "workers/worker.py": [
        ('backend="nccl"', 'backend="hccl"'),
        ('{"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node}',
         '{"NPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node}'),
    ],
    "inference_engines/vllm/vllm_engine.py": [
        ('backend="nccl"', 'backend="hccl"'),
        ('device="cuda"', 'device="npu"'),
        ('torch.cuda.current_device()', 'torch.npu.current_device()'),
        ('torch.cuda.get_device_properties', 'torch.npu.get_device_properties'),
    ],
    "model_wrapper.py": [
        ('from flash_attn.bert_padding import pad_input, unpad_input',
         'try:\n    from flash_attn.bert_padding import pad_input, unpad_input\n'
         'except ImportError:\n    pad_input = None\n    unpad_input = None'),
    ],
    "utils/utils.py": [],  # proxy forwarding added manually
}

print(f"Patching {base} for NPU...")
for relpath, replacements in patches.items():
    fpath = os.path.join(base, relpath)
    if not os.path.exists(fpath):
        print(f"  SKIP {relpath} (not found)")
        continue
    with open(fpath, 'r') as f:
        content = f.read()
    changed = 0
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changed += 1
    with open(fpath, 'w') as f:
        f.write(content)
    print(f"  {relpath}: {changed}/{len(replacements)} replacements")
print("Done!")
