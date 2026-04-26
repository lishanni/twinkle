# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import os
import sys
import types
import unittest
from unittest import mock

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
