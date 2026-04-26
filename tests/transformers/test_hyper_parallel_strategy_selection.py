# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys

# Prevent torch from auto-loading torch_npu as a device backend, which would
# cause "Two accelerators: npu and npu" when transformers later imports it.
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import numpy as np
import types
import unittest
from unittest import mock

# Import torch and torch_npu BEFORE transformers/peft to avoid double-registration
import torch
if hasattr(torch, 'npu') and hasattr(torch.npu, 'is_available') and torch.npu.is_available():
    import torch_npu  # noqa: F401

_original_sysconf = os.sysconf


def _patched_sysconf(name):
    if name == 'SC_SEM_NSEMS_MAX':
        return 256
    return _original_sysconf(name)


os.sysconf = _patched_sysconf
if 'zmq' not in sys.modules:
    fake_zmq = types.ModuleType('zmq')
    fake_zmq.Socket = type('Socket', (), {})
    fake_zmq.SNDMORE = 1
    fake_zmq.PUB = 1
    fake_zmq.SUB = 2
    fake_zmq.SUBSCRIBE = 6
    fake_zmq.IPV6 = 42
    fake_zmq.Context = lambda: None
    sys.modules['zmq'] = fake_zmq

import twinkle.model.transformers.transformers as transformers_module
from twinkle.model.transformers.strategy.hyper_parallel import _HyperParallelArguments
from twinkle.utils import DeviceMesh


class _DummyAccelerateStrategy:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def pretrained_load_context(self):
        return mock.MagicMock()


class _DummyNativeFSDPStrategy:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def pretrained_load_context(self):
        return mock.MagicMock()


class _DummyHyperParallelStrategy:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def pretrained_load_context(self):
        return mock.MagicMock()


class TestHyperParallelArguments(unittest.TestCase):

    def test_from_config_accepts_known_fields_only(self):
        hp_args = _HyperParallelArguments.from_config({
            'tp_size': 1,
            'param_dtype': 'bf16',
            'reduce_dtype': 'bf16',
            'unknown_field': 'ignored',
        })
        self.assertEqual(hp_args.tp_size, 1)
        self.assertEqual(hp_args.param_dtype, 'bf16')
        self.assertEqual(hp_args.reduce_dtype, 'bf16')

    def test_from_config_rejects_non_fsdp_tp_size(self):
        with self.assertRaises(ValueError):
            _HyperParallelArguments.from_config({'tp_size': 2})


class TestTransformersModelStrategySelection(unittest.TestCase):

    @staticmethod
    def _build_model_stub(
        *,
        use_hyper_parallel: bool,
        strategy: str = 'accelerate',
        fsdp_config=None,
        device_mesh=None,
    ):
        model = transformers_module.TransformersModel.__new__(transformers_module.TransformersModel)
        model.mixed_precision = 'bf16'
        model.device_mesh = device_mesh
        model._fsdp_config = dict(fsdp_config or {})
        model._ddp_config = {'bucket_cap_mb': 32}
        model._use_hyper_parallel = use_hyper_parallel or strategy == 'hyper_parallel'
        model._hyper_parallel_config = {'tp_size': 1, 'param_dtype': 'bf16'}
        model._memory_efficient_init = True
        return model

    def test_use_hyper_parallel_switches_to_hyper_parallel_strategy(self):
        model = self._build_model_stub(use_hyper_parallel=True)
        with mock.patch.object(
                transformers_module, 'HyperParallelStrategy', _DummyHyperParallelStrategy), mock.patch.object(
                    transformers_module, 'AccelerateStrategy', _DummyAccelerateStrategy), mock.patch.object(
                        transformers_module, 'NativeFSDPStrategy', _DummyNativeFSDPStrategy):
            transformers_module.TransformersModel._decide_strategy(model, strategy='accelerate')

        self.assertIsInstance(model.strategy, _DummyHyperParallelStrategy)
        self.assertEqual(model.strategy.kwargs.get('hyper_parallel_config'), {'tp_size': 1, 'param_dtype': 'bf16'})

    def test_use_hyper_parallel_rejects_expert_parallel(self):
        device_mesh = DeviceMesh(
            mesh=np.arange(8).reshape(2, 4),
            mesh_dim_names=('dp', 'fsdp'),
            device_type='npu',
            ep_size=2,
        )
        model = self._build_model_stub(
            use_hyper_parallel=True,
            fsdp_config={
                'expert_parallel': {
                    'enabled': True,
                    'ep_size': 2,
                }
            },
            device_mesh=device_mesh,
        )
        with mock.patch.object(transformers_module, 'HyperParallelStrategy', _DummyHyperParallelStrategy):
            with self.assertRaises(NotImplementedError):
                transformers_module.TransformersModel._decide_strategy(model, strategy='accelerate')

    def test_accelerate_strategy_remains_default_without_switch(self):
        model = self._build_model_stub(use_hyper_parallel=False)
        with mock.patch.object(
                transformers_module, 'HyperParallelStrategy', _DummyHyperParallelStrategy), mock.patch.object(
                    transformers_module, 'AccelerateStrategy', _DummyAccelerateStrategy), mock.patch.object(
                        transformers_module, 'NativeFSDPStrategy', _DummyNativeFSDPStrategy):
            transformers_module.TransformersModel._decide_strategy(model, strategy='accelerate')

        self.assertIsInstance(model.strategy, _DummyAccelerateStrategy)


class TestStrategyInterfaceConformance(unittest.TestCase):
    """Verify all three strategy classes implement the unified interface methods."""

    INTERFACE_METHODS = [
        'pretrained_load_context',
        'wrap_model',
        'unwrap_model',
        'get_full_state_dict',
        'wrap_optimizer',
        'adjust_optimizer_kwargs',
        'get_ep_clip_kwargs',
        'log_device_memory',
    ]

    def test_accelerate_strategy_has_all_interface_methods(self):
        from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy
        for method_name in self.INTERFACE_METHODS:
            self.assertTrue(
                hasattr(AccelerateStrategy, method_name),
                f'AccelerateStrategy missing interface method: {method_name}',
            )

    def test_native_fsdp_strategy_has_all_interface_methods(self):
        from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy
        for method_name in self.INTERFACE_METHODS:
            self.assertTrue(
                hasattr(NativeFSDPStrategy, method_name),
                f'NativeFSDPStrategy missing interface method: {method_name}',
            )

    def test_hyper_parallel_strategy_has_all_interface_methods(self):
        from twinkle.model.transformers.strategy.hyper_parallel import HyperParallelStrategy
        for method_name in self.INTERFACE_METHODS:
            self.assertTrue(
                hasattr(HyperParallelStrategy, method_name),
                f'HyperParallelStrategy missing interface method: {method_name}',
            )
        # Deprecated alias should still exist
        self.assertTrue(
            hasattr(HyperParallelStrategy, 'log_npu_memory'),
            'HyperParallelStrategy missing deprecated alias: log_npu_memory',
        )

    def test_hyper_parallel_inherits_from_accelerate(self):
        from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy
        from twinkle.model.transformers.strategy.hyper_parallel import HyperParallelStrategy
        self.assertTrue(
            issubclass(HyperParallelStrategy, AccelerateStrategy),
            'HyperParallelStrategy should inherit from AccelerateStrategy',
        )


class TestAccelerateStrategyNoop(unittest.TestCase):
    """Verify AccelerateStrategy base methods have correct no-op defaults."""

    def test_wrap_optimizer_passes_through(self):
        # We can't instantiate AccelerateStrategy without a real Accelerator,
        # so test the method signature and default behavior via mock.
        strategy = mock.MagicMock(spec=[])
        # Simulate the no-op: returns the optimizer as-is
        from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy
        result = AccelerateStrategy.wrap_optimizer(strategy, 'fake_optimizer')
        self.assertEqual(result, 'fake_optimizer')

    def test_adjust_optimizer_kwargs_passes_through(self):
        from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy
        strategy = mock.MagicMock(spec=[])
        kwargs = {'lr': 1e-4, 'weight_decay': 0.01}
        result = AccelerateStrategy.adjust_optimizer_kwargs(strategy, 'AdamW', kwargs)
        self.assertEqual(result, {'lr': 1e-4, 'weight_decay': 0.01})

    def test_get_ep_clip_kwargs_returns_empty(self):
        from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy
        strategy = mock.MagicMock(spec=[])
        result = AccelerateStrategy.get_ep_clip_kwargs(strategy, None)
        self.assertEqual(result, {})

    def test_log_device_memory_is_noop(self):
        from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy
        strategy = mock.MagicMock(spec=[])
        logger = mock.MagicMock()
        # Should not raise
        AccelerateStrategy.log_device_memory(strategy, None, logger, 'test')
        logger.info.assert_not_called()


class TestSetOptimizerDelegation(unittest.TestCase):
    """Verify set_optimizer delegates to strategy.wrap_optimizer instead of direct HP import."""

    def test_set_optimizer_calls_strategy_wrap_optimizer(self):
        model = transformers_module.TransformersModel.__new__(transformers_module.TransformersModel)
        model._use_hyper_parallel = False
        model._fsdp_config = {}
        model._ddp_config = {}
        model.device_mesh = None
        model.mixed_precision = 'bf16'
        model._memory_efficient_init = False
        model.strategy = mock.MagicMock()
        model.strategy.adjust_optimizer_kwargs.return_value = {'lr': 1e-4}
        model.strategy.wrap_optimizer.return_value = None
        model.optimizer_group = {
            'default': transformers_module.OptimizerGroup(
                optimizer=None,
                lr_scheduler=None,
            ),
        }
        model.active_group = 'default'
        model.optimizer_group['default'].adapter_name = 'default'

        # Mock _create_param_group to return a real tensor so construct_class works
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        model._create_param_group = mock.MagicMock(return_value=[dummy_param])

        model.set_optimizer(optimizer_cls='AdamW', lr=1e-4)

        model.strategy.adjust_optimizer_kwargs.assert_called_once()
        model.strategy.wrap_optimizer.assert_called_once()


if __name__ == '__main__':
    unittest.main()
