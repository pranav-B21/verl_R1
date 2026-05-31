# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# MODIFIED: CPU round-trip replaces CUDA IPC so pidfd_getfd is never called.
# Needed on TACC nodes where Singularity's seccomp policy blocks pidfd_getfd.
# Drop-in replacement for sglang/srt/patch_torch.py.

import io
from typing import Callable, Union

import torch
from packaging import version
from torch.multiprocessing import reductions


def monkey_patch_torch_reductions():
    """CPU-based override: avoids CUDA IPC (pidfd_getfd) blocked by Singularity seccomp."""
    if hasattr(reductions, "_reduce_tensor_original"):
        return

    reductions._reduce_tensor_original = reductions.reduce_tensor
    reductions._rebuild_cuda_tensor_original = reductions.rebuild_cuda_tensor

    reductions.reduce_tensor = _reduce_tensor_modified
    reductions.rebuild_cuda_tensor = _rebuild_cuda_tensor_modified

    reductions.init_reductions()


def _rebuild_cuda_tensor_from_cpu(buf_bytes, device_str, requires_grad):
    """Rebuild a CUDA tensor from CPU-serialized bytes (no CUDA IPC needed)."""
    buf = io.BytesIO(buf_bytes)
    cpu_tensor = torch.load(buf, map_location="cpu", weights_only=True)
    tensor = cpu_tensor.to(device_str, non_blocking=False)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def _reduce_tensor_modified(tensor):
    if tensor.is_cuda:
        # Copy to CPU and serialize as raw bytes — no CUDA IPC handle created.
        device_str = str(tensor.device)
        requires_grad = tensor.requires_grad
        cpu_tensor = tensor.detach().contiguous().cpu()
        buf = io.BytesIO()
        torch.save(cpu_tensor, buf)
        return (_rebuild_cuda_tensor_from_cpu, (buf.getvalue(), device_str, requires_grad))
    return reductions._reduce_tensor_original(tensor)


# _REDUCE_TENSOR_ARG_DEVICE_INDEX kept for _rebuild_cuda_tensor_modified below.
_REDUCE_TENSOR_ARG_DEVICE_INDEX = 6


def _rebuild_cuda_tensor_modified(*args):
    # Only reached if data was serialized before this patch was active (shouldn't
    # happen in normal operation). Apply the original UUID conversion and attempt
    # the CUDA IPC path — will fail with pidfd_getfd on strict nodes.
    args = _modify_tuple(args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_from_maybe_uuid)
    return reductions._rebuild_cuda_tensor_original(*args)


def _device_to_uuid(device: int) -> str:
    return str(torch.cuda.get_device_properties(device).uuid)


def _device_from_maybe_uuid(device_maybe_uuid: Union[int, str]) -> int:
    if isinstance(device_maybe_uuid, int):
        return device_maybe_uuid

    if isinstance(device_maybe_uuid, str):
        for device in range(torch.cuda.device_count()):
            if str(torch.cuda.get_device_properties(device).uuid) == device_maybe_uuid:
                return device
        raise Exception("Invalid device_uuid=" + device_maybe_uuid)

    raise Exception(f"Unknown type: {device_maybe_uuid=}")


def _modify_tuple(t, index: int, modifier: Callable):
    return *t[:index], modifier(t[index]), *t[index + 1:]


def monkey_patch_torch_compile():
    if version.parse(torch.__version__) < version.parse("2.8.0"):
        # These things are cacheable by torch.compile. torch.compile just doesn't know it.
        # This was fixed in PyTorch 2.8, but until then, we monkey patch.
        import torch._higher_order_ops.auto_functionalize as af

        af.auto_functionalized_v2._cacheable = True
        af.auto_functionalized._cacheable = True
