"""Profile native FSDP2 vs HyperParallel FSDP2 on Qwen3-32B — single-step profiling at step 10.

Usage:
    torchrun --nproc_per_node=4 cookbook/transformers/profile_fsdp2_compare_32b.py --backend native
    torchrun --nproc_per_node=4 cookbook/transformers/profile_fsdp2_compare_32b.py --backend hp
"""
import argparse
import os

from peft import LoraConfig

import torch
import torch_npu.profiler as prof
import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['native', 'hp'], required=True)
parser.add_argument('--max-length', type=int, default=2048,
                    help='Max sequence length (default 2048)')
args = parser.parse_args()

world_size = torch.npu.device_count()
device_mesh = DeviceMesh.from_sizes(fsdp_size=world_size, dp_size=1, device_type='npu')
twinkle.initialize(mode='local', nproc_per_node=world_size, global_device_mesh=device_mesh)

logger = get_logger()

MODEL_ID = 'ms://Qwen/Qwen3-32B'
DATASET_ID = 'ms://swift/self-cognition'

PROF_DIR = os.path.join(
    os.path.dirname(__file__),
    f'../../log/profiling/32b_{"hp" if args.backend == "hp" else "native"}_fsdp2',
)

WARMUP_STEPS = 10


def log_memory(tag: str = '') -> None:
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if hasattr(torch, 'npu') and torch.npu.is_available():
        allocated = torch.npu.memory_allocated(local_rank) / (1024 ** 3)
        reserved = torch.npu.memory_reserved(local_rank) / (1024 ** 3)
        peak = torch.npu.max_memory_allocated(local_rank) / (1024 ** 3)
        prefix = f'[{tag}] ' if tag else ''
        logger.info(
            f'{prefix}NPU {local_rank} mem: '
            f'alloc={allocated:.2f}GB, reserved={reserved:.2f}GB, peak={peak:.2f}GB'
        )


def train():
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(1000)))
    dataset.set_template('Template', model_id=MODEL_ID, max_length=args.max_length)
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=4)

    if args.backend == 'hp':
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=device_mesh,
            use_hyper_parallel=True,
            hyper_parallel_config={
                'tp_size': 1,
                'device_type': 'npu',
                'param_dtype': 'bf16',
                'reduce_dtype': 'bf16',
                'reshard_after_forward': True,
                'comm_fusion': False,
                'enable_prefetch': True,
                'num_forward_prefetch': 1,
                'num_backward_prefetch': 1,
                'upcast_trainable_params': True,
            },
        )
    else:
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=device_mesh,
        )
    model.model._no_split_modules = {'Qwen3DecoderLayer'}

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4)
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=5,
        num_training_steps=len(dataloader),
    )

    backend_label = 'HyperParallel FSDP2' if args.backend == 'hp' else 'native FSDP2'
    logger.info(f'Backend: {backend_label} (Qwen3-32B, profiling step {WARMUP_STEPS} only)')
    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    log_memory('after model init')

    # Warmup steps 0–9
    dataloader_iter = iter(dataloader)
    logger.info(f'Warmup steps 0–{WARMUP_STEPS - 1}...')
    for step in range(WARMUP_STEPS):
        batch = next(dataloader_iter)
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        if step % 5 == 0:
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Warmup step {step}/{WARMUP_STEPS - 1}, metric: {metric}')
            log_memory(f'warmup step {step}')

    # Profile step 10 only
    os.makedirs(PROF_DIR, exist_ok=True)
    experimental_config = prof._ExperimentalConfig(profiler_level=prof.ProfilerLevel.Level1)
    with prof.profile(
        activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
        with_stack=True,
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=prof.tensorboard_trace_handler(PROF_DIR),
        experimental_config=experimental_config,
    ) as p:
        batch = next(dataloader_iter)
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        p.step()
        logger.info(f'Profile step {WARMUP_STEPS} done')

    logger.info(f'Profiling data saved to {PROF_DIR}')

    # Continue remaining steps (not profiled)
    logger.info(f'Continuing steps {WARMUP_STEPS + 1}–{len(dataloader) - 1}...')
    for step in range(WARMUP_STEPS + 1, len(dataloader)):
        batch = next(dataloader_iter)
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        if step % 20 == 0:
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Step {step}/{len(dataloader) - 1}, metric: {metric}')
            log_memory(f'step {step}')

    logger.info('Training complete.')
    log_memory('final')


if __name__ == '__main__':
    train()
