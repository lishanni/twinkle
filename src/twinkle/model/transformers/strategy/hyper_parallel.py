# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import types
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch

from twinkle import DeviceMesh
from twinkle.utils import exists

from .accelerate import AccelerateStrategy

_VALID_DTYPES = {'float32', 'float16', 'bfloat16', 'fp32', 'fp16', 'bf16'}


@dataclass
class _HyperParallelArguments:
    """HyperParallel strategy arguments."""

    tp_size: int = 1
    device_type: str = 'auto'
    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    reshard_after_forward: Optional[bool] = None
    # Performance optimization
    comm_fusion: bool = True
    enable_prefetch: bool = True
    num_forward_prefetch: int = 1
    num_backward_prefetch: int = 1
    # Precision control
    upcast_trainable_params: bool = True

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> '_HyperParallelArguments':
        known_fields = set(cls.__dataclass_fields__)  # pylint: disable=no-member
        hp_args = cls(**{k: v for k, v in config.items() if k in known_fields})
        hp_args.validate()
        return hp_args

    def validate(self) -> None:
        if self.tp_size != 1:
            raise ValueError(
                'Current HyperParallel integration in Twinkle only supports FSDP/fully_shard. '
                f'Expected tp_size=1, got {self.tp_size}.'
            )
        if self.device_type not in {'auto', 'cpu', 'cuda', 'npu'}:
            raise ValueError(
                f"device_type must be one of ['auto', 'cpu', 'cuda', 'npu'], got {self.device_type!r}."
            )
        if self.param_dtype is not None and self.param_dtype not in _VALID_DTYPES:
            raise ValueError(f'Unsupported param_dtype={self.param_dtype!r}.')
        if self.reduce_dtype is not None and self.reduce_dtype not in _VALID_DTYPES:
            raise ValueError(f'Unsupported reduce_dtype={self.reduce_dtype!r}.')
        if self.reshard_after_forward is not None and not isinstance(self.reshard_after_forward, bool):
            raise ValueError('reshard_after_forward must be a bool when provided.')
        if not isinstance(self.comm_fusion, bool):
            raise ValueError(f'comm_fusion must be a bool, got {type(self.comm_fusion).__name__}.')
        if not isinstance(self.enable_prefetch, bool):
            raise ValueError(f'enable_prefetch must be a bool, got {type(self.enable_prefetch).__name__}.')
        if not isinstance(self.upcast_trainable_params, bool):
            raise ValueError(f'upcast_trainable_params must be a bool, got {type(self.upcast_trainable_params).__name__}.')


def _wrap_optimizer_step_with_skip_dtensor_dispatch(optimizer) -> None:
    """Wrap optimizer.step so DTensor dispatch is skipped during updates."""
    if optimizer is None or getattr(optimizer, '_hp_step_wrapped', False):
        return

    from hyper_parallel import SkipDTensorDispatch

    original_step = optimizer.step

    def _hp_step(bound_optimizer, *args, **kwargs):
        del bound_optimizer
        with SkipDTensorDispatch():
            return original_step(*args, **kwargs)

    optimizer.step = types.MethodType(_hp_step, optimizer)
    setattr(optimizer, '_hp_step_wrapped', True)


def _normalize_full_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize full state dict to fp32 for stable checkpoint compatibility."""
    normalized: dict[str, Any] = {}
    cast_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            normalized[key] = value
            continue

        if not torch.is_floating_point(value) or value.dtype == torch.float32:
            normalized[key] = value.cpu()
            continue

        storage = value.untyped_storage()
        cache_key = (
            storage.data_ptr(),
            value.storage_offset(),
            tuple(value.size()),
            tuple(value.stride()),
            value.device.type,
            str(value.dtype),
        )
        casted = cast_cache.get(cache_key)
        if casted is None:
            casted = value.to(dtype=torch.float32, device='cpu')
            cast_cache[cache_key] = casted
        normalized[key] = casted
    return normalized


class HyperParallelStrategy(AccelerateStrategy):
    """Accelerate strategy that swaps torch FSDP2 wrapping with HyperParallel fully_shard."""

    def __init__(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
        ddp_config: Dict[str, Any] = None,
        fsdp_config: Dict[str, Any] = None,
        memory_efficient_init: bool = False,
        hyper_parallel_config: Dict[str, Any] = None,
    ):
        if not exists('hyper_parallel'):
            raise ImportError(
                "hyper_parallel is not installed. Install it first: `pip install hyper_parallel`."
            )
        self._hp_args = _HyperParallelArguments.from_config(hyper_parallel_config or {})
        super().__init__(
            device_mesh=device_mesh,
            mixed_precision=mixed_precision,
            ddp_config=ddp_config,
            fsdp_config=fsdp_config,
            memory_efficient_init=memory_efficient_init,
        )

    def wrap_model(self, model, *args):
        import accelerate.accelerator as acc_module
        from hyper_parallel.integration.llamafactory.utils import fsdp2_prepare_model as hp_fsdp2_prepare_model

        origin_fsdp2_prepare_model = acc_module.fsdp2_prepare_model

        def _hp_fsdp2_prepare_model(accelerator, wrapped_model):
            return hp_fsdp2_prepare_model(accelerator, wrapped_model, self._hp_args)

        acc_module.fsdp2_prepare_model = _hp_fsdp2_prepare_model
        try:
            wrapped = super().wrap_model(model, *args)
        finally:
            acc_module.fsdp2_prepare_model = origin_fsdp2_prepare_model

        if isinstance(wrapped, tuple) and len(wrapped) > 1:
            optimizer = wrapped[1]
            _wrap_optimizer_step_with_skip_dtensor_dispatch(optimizer)
        return wrapped

    def get_full_state_dict(self, model) -> dict:
        """Collect full state dict from HyperParallel FSDP2 model."""
        from hyper_parallel.core.fully_shard.api import get_model_state_dict as hp_get_model_state_dict
        from torch.distributed.checkpoint.state_dict import StateDictOptions

        unwrapped = self.unwrap_model(model)
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = hp_get_model_state_dict(unwrapped, options=options)
        return _normalize_full_state_dict(state_dict)

    def log_npu_memory(self, model, logger, tag: str = '') -> None:
        """Log NPU memory usage for current rank."""
        import torch
        if not (hasattr(torch, 'npu') and torch.npu.is_available()):
            return
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        allocated = torch.npu.memory_allocated(local_rank) / (1024 ** 3)
        reserved = torch.npu.memory_reserved(local_rank) / (1024 ** 3)
        max_allocated = torch.npu.max_memory_allocated(local_rank) / (1024 ** 3)
        prefix = f'[{tag}] ' if tag else ''
        logger.info(
            f'{prefix}NPU {local_rank} memory: '
            f'allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, peak={max_allocated:.2f}GB'
        )
