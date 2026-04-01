"""
Monkey-patch torch.cuda → torch.npu for running CUDA-only code on Ascend NPU.

Import this module ONCE at the very start of your program, before any other
torch imports:

    import npu_support.patch_cuda   # noqa: F401  — must be first!
    import torch
    ...

After this, all code that calls torch.cuda.* will transparently use NPU.
This includes PyTorch internals (FSDP, DeviceMesh, etc.).

The patch is skipped if real CUDA is available, so this is safe to import
unconditionally.
"""

import importlib
import logging
import os
import sys
import types

logger = logging.getLogger("npu_support.patch_cuda")

_PATCHED = False


def _should_patch():
    """Decide whether to apply the NPU patch."""
    if os.environ.get("SKYRL_NO_NPU_PATCH", "0") == "1":
        return False
    try:
        import torch
        if torch.cuda.is_available():
            return False  # Real CUDA present, no patching needed
    except Exception:
        pass
    try:
        import torch_npu  # noqa: F401
        import torch
        return hasattr(torch, "npu") and torch.npu.is_available()
    except ImportError:
        return False


class _CudaProxy(types.ModuleType):
    """
    A module proxy that forwards attribute access from torch.cuda to torch.npu.

    This is the key trick: we replace sys.modules["torch.cuda"] with this proxy.
    Any code doing `import torch.cuda` or `torch.cuda.xxx()` will hit __getattr__
    here and get redirected to torch.npu.

    We keep a reference to the original torch.cuda so we can fall back for
    attributes that don't exist on torch.npu (e.g. documentation strings).
    """

    def __init__(self, orig_cuda, npu_mod):
        super().__init__("torch.cuda")
        self._orig = orig_cuda
        self._npu = npu_mod
        # Copy over module metadata
        self.__package__ = getattr(orig_cuda, "__package__", "torch.cuda")
        self.__path__ = getattr(orig_cuda, "__path__", [])
        self.__file__ = getattr(orig_cuda, "__file__", "")
        self.__loader__ = getattr(orig_cuda, "__loader__", None)
        self.__spec__ = getattr(orig_cuda, "__spec__", None)

    def __getattr__(self, name):
        # Priority: npu module first, then original cuda as fallback
        if hasattr(self._npu, name):
            return getattr(self._npu, name)
        if hasattr(self._orig, name):
            return getattr(self._orig, name)
        raise AttributeError(f"module 'torch.cuda' (proxied to torch.npu) has no attribute '{name}'")


class _NPUDeviceProperties:
    """Stub for torch.cuda.get_device_properties() on NPU."""
    def __init__(self, idx):
        self.name = f"Ascend NPU {idx}"
        self.uuid = f"NPU-{idx}"
        self.major = 9
        self.minor = 0
        self.total_memory = 64 * 1024 * 1024 * 1024  # 64 GB HBM
        self.multi_processor_count = 32


def _apply_patch():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    import torch
    import torch_npu  # noqa: F401

    orig_cuda = torch.cuda
    npu = torch.npu

    # --- 1. Replace torch.cuda module with our proxy ---
    proxy = _CudaProxy(orig_cuda, npu)

    # Override specific functions that need special handling
    proxy.is_available = lambda: True
    proxy.device_count = npu.device_count
    proxy.current_device = npu.current_device
    proxy.set_device = npu.set_device
    proxy.empty_cache = npu.empty_cache
    proxy.synchronize = npu.synchronize
    proxy.manual_seed = npu.manual_seed
    proxy.manual_seed_all = npu.manual_seed_all
    proxy.memory_allocated = npu.memory_allocated
    proxy.max_memory_allocated = npu.max_memory_allocated

    if hasattr(npu, "get_rng_state"):
        proxy.get_rng_state = npu.get_rng_state
    if hasattr(npu, "set_rng_state"):
        proxy.set_rng_state = npu.set_rng_state
    if hasattr(npu, "mem_get_info"):
        proxy.mem_get_info = npu.mem_get_info

    def _get_device_properties(device=None):
        if device is None:
            device = npu.current_device()
        if isinstance(device, torch.device):
            device = device.index or 0
        return _NPUDeviceProperties(device)
    proxy.get_device_properties = _get_device_properties

    def _ipc_collect():
        pass  # No-op on NPU
    proxy.ipc_collect = _ipc_collect

    # Proxy the memory sub-module too
    if hasattr(orig_cuda, "memory"):
        mem_proxy = types.SimpleNamespace()
        if hasattr(npu, "mem_get_info"):
            mem_proxy.mem_get_info = npu.mem_get_info
        proxy.memory = mem_proxy

    # Install the proxy
    sys.modules["torch.cuda"] = proxy
    torch.cuda = proxy

    # --- 1b. Patch Tensor.cuda() → Tensor.npu() ---
    _orig_tensor_cuda = torch.Tensor.cuda

    def _tensor_cuda(self, device=None, non_blocking=False, **kwargs):
        if device is None:
            return self.npu(npu.current_device(), non_blocking=non_blocking)
        if isinstance(device, int):
            return self.npu(device, non_blocking=non_blocking)
        return self.npu(non_blocking=non_blocking)

    torch.Tensor.cuda = _tensor_cuda

    # --- 2. Patch init_device_mesh to use "npu" instead of "cuda" ---
    try:
        from torch.distributed.device_mesh import init_device_mesh as _orig_init_device_mesh

        def _patched_init_device_mesh(device_type, *args, **kwargs):
            if device_type == "cuda":
                device_type = "npu"
            return _orig_init_device_mesh(device_type, *args, **kwargs)

        import torch.distributed.device_mesh
        torch.distributed.device_mesh.init_device_mesh = _patched_init_device_mesh
        # Also patch the commonly used import path
        if hasattr(torch.distributed, "init_device_mesh"):
            torch.distributed.init_device_mesh = _patched_init_device_mesh
    except ImportError:
        pass

    # --- 3. Patch flash_attn imports (not available on NPU) ---
    _install_flash_attn_stub()

    # --- 4. Patch Ray to use NPU resources instead of GPU ---
    _patch_ray_for_npu()

    # --- 4b. Patch torch.distributed to use HCCL instead of NCCL ---
    _orig_init_pg = torch.distributed.init_process_group

    def _patched_init_process_group(*args, **kwargs):
        if kwargs.get("backend") == "nccl":
            kwargs["backend"] = "hccl"
        elif len(args) > 0 and args[0] == "nccl":
            args = ("hccl",) + args[1:]
        return _orig_init_pg(*args, **kwargs)

    torch.distributed.init_process_group = _patched_init_process_group

    # Also patch the StatelessProcessGroup used by vllm weight sync
    try:
        from torch.distributed import new_group as _orig_new_group

        def _patched_new_group(*args, **kwargs):
            if kwargs.get("backend") == "nccl":
                kwargs["backend"] = "hccl"
            return _orig_new_group(*args, **kwargs)

        torch.distributed.new_group = _patched_new_group
    except Exception:
        pass

    # --- 5. Set NPU-friendly environment defaults ---
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "360")
    os.environ.setdefault("HCCL_EXEC_TIMEOUT", "360")

    logger.info("NPU patch applied: torch.cuda → torch.npu, DeviceMesh → npu, flash_attn stubbed, Ray GPU→NPU")


def _patch_ray_for_npu():
    """Patch Ray to treat NPU as the accelerator resource.

    Ray doesn't natively map NPU to GPU. We patch:
    1. ray.remote() — convert num_gpus=N to resources={"NPU": N}
    2. The placement_group function — convert {"GPU": N} bundles to {"NPU": N}
    """
    try:
        import ray
    except ImportError:
        return

    # --- Patch ray.remote to convert num_gpus → NPU resource ---
    _orig_remote = ray.remote

    def _patched_remote(*args, **kwargs):
        if "num_gpus" in kwargs and kwargs["num_gpus"]:
            n = kwargs.pop("num_gpus")
            resources = kwargs.get("resources", {}) or {}
            resources["NPU"] = n
            kwargs["resources"] = resources
        result = _orig_remote(*args, **kwargs)
        # Also patch the .options() method on the returned RemoteFunction/ActorClass
        if hasattr(result, "options"):
            _orig_options = result.options

            def _patched_options(**opt_kwargs):
                if "num_gpus" in opt_kwargs and opt_kwargs["num_gpus"]:
                    n = opt_kwargs.pop("num_gpus")
                    resources = opt_kwargs.get("resources", {}) or {}
                    resources["NPU"] = n
                    opt_kwargs["resources"] = resources
                return _orig_options(**opt_kwargs)

            result.options = _patched_options
        return result

    ray.remote = _patched_remote

    # --- Also patch ActorClass.options and RemoteFunction.options ---
    # This catches `SomeActor.options(num_gpus=1).remote(...)` patterns
    try:
        from ray.actor import ActorClass
        _orig_actor_options = ActorClass.options

        def _patched_actor_options(self, **opt_kwargs):
            if "num_gpus" in opt_kwargs and opt_kwargs["num_gpus"]:
                n = opt_kwargs.pop("num_gpus")
                resources = opt_kwargs.get("resources", {}) or {}
                resources["NPU"] = n
                opt_kwargs["resources"] = resources
            return _orig_actor_options(self, **opt_kwargs)

        ActorClass.options = _patched_actor_options
    except Exception:
        pass

    try:
        from ray.remote_function import RemoteFunction
        _orig_func_options = RemoteFunction.options

        def _patched_func_options(self, **opt_kwargs):
            if "num_gpus" in opt_kwargs and opt_kwargs["num_gpus"]:
                n = opt_kwargs.pop("num_gpus")
                resources = opt_kwargs.get("resources", {}) or {}
                resources["NPU"] = n
                opt_kwargs["resources"] = resources
            return _orig_func_options(self, **opt_kwargs)

        RemoteFunction.options = _patched_func_options
    except Exception:
        pass

    # --- Patch placement_group to convert GPU → NPU in bundles ---
    # The function lives in ray.util.placement_group module
    try:
        import importlib
        _pg_module = importlib.import_module("ray.util.placement_group")
        _orig_pg = _pg_module.placement_group

        def _patched_placement_group(bundles, *args, **kwargs):
            patched_bundles = []
            for bundle in bundles:
                new_bundle = {}
                for k, v in bundle.items():
                    if k == "GPU":
                        new_bundle["NPU"] = v
                    else:
                        new_bundle[k] = v
                patched_bundles.append(new_bundle)
            return _orig_pg(patched_bundles, *args, **kwargs)

        _pg_module.placement_group = _patched_placement_group
    except Exception as e:
        logger.warning(f"Failed to patch placement_group: {e}")


def _install_flash_attn_stub():
    """Install a stub for flash_attn so imports don't crash."""
    if "flash_attn" in sys.modules:
        return  # Already imported (maybe real flash_attn is available)
    try:
        import flash_attn  # noqa: F401
        return  # Real flash_attn available, don't stub
    except ImportError:
        pass

    def _make_stub(name, is_package=False):
        mod = types.ModuleType(name)
        mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        # Give it a proper __spec__ so importlib.util.find_spec() doesn't choke
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=is_package)
        if is_package:
            mod.__path__ = []
        mod.__version__ = "0.0.0"  # stub version
        return mod

    flash_attn_mod = _make_stub("flash_attn", is_package=True)

    bert_padding = _make_stub("flash_attn.bert_padding")

    def _pad_input(hidden_states, indices, batch, seqlen):
        raise NotImplementedError("flash_attn.bert_padding.pad_input is not available on NPU")

    def _unpad_input(hidden_states, attention_mask):
        raise NotImplementedError("flash_attn.bert_padding.unpad_input is not available on NPU")

    bert_padding.pad_input = _pad_input
    bert_padding.unpad_input = _unpad_input

    ops_mod = _make_stub("flash_attn.ops", is_package=True)
    triton_mod = _make_stub("flash_attn.ops.triton", is_package=True)
    ce_mod = _make_stub("flash_attn.ops.triton.cross_entropy")

    sys.modules["flash_attn"] = flash_attn_mod
    sys.modules["flash_attn.bert_padding"] = bert_padding
    sys.modules["flash_attn.ops"] = ops_mod
    sys.modules["flash_attn.ops.triton"] = triton_mod
    sys.modules["flash_attn.ops.triton.cross_entropy"] = ce_mod


# --- Auto-apply on import ---
if _should_patch():
    _apply_patch()
