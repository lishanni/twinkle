"""Qwen3-0.6B deterministic comparison: native FSDP2 vs HyperParallel FSDP2.

Supports both full-tuning (default) and LoRA (--use-lora) modes.

Usage:
    # Native FSDP2
    ASCEND_RT_VISIBLE_DEVICES=0,1 \\
    ASCEND_LAUNCH_BLOCKING=1 HCCL_DETERMINISTIC=true \\
    torchrun --nproc_per_node=2 cookbook/transformers/compare_fsdp2_fulltune_06b.py \\
        --backend native --gradient-accumulation-steps 2

    # HyperParallel FSDP2
    ASCEND_RT_VISIBLE_DEVICES=2,3 \\
    ASCEND_LAUNCH_BLOCKING=1 HCCL_DETERMINISTIC=true \\
    torchrun --nproc_per_node=2 --master-port 29501 \\
    cookbook/transformers/compare_fsdp2_fulltune_06b.py \\
        --backend hp --gradient-accumulation-steps 2
"""
import argparse
import os

import torch
import twinkle

# Deterministic computation for reproducible loss comparison
torch.manual_seed(42)
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['native', 'hp'], required=True)
parser.add_argument('--gradient-accumulation-steps', type=int, default=2,
                    help='Gradient accumulation steps (default 2)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default 1e-4)')
parser.add_argument('--max-steps', type=int, default=None,
                    help='Stop after N steps (default: full epoch)')
parser.add_argument('--use-lora', action='store_true',
                    help='Use LoRA instead of full-tuning (r=8, all-linear)')
args = parser.parse_args()

world_size = torch.npu.device_count()
device_mesh = DeviceMesh.from_sizes(fsdp_size=world_size, dp_size=1, device_type='npu')
twinkle.initialize(mode='local', nproc_per_node=world_size, global_device_mesh=device_mesh)

logger = get_logger()

MODEL_ID = 'ms://Qwen/Qwen3-0.6B'
DATASET_ID = 'ms://swift/self-cognition'


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
    dataset.set_template('Template', model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=8)

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
            },
        )
    else:
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=device_mesh,
        )

    model.model._no_split_modules = {'Qwen3DecoderLayer'}

    if args.use_lora:
        from peft import LoraConfig
        lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')
        model.add_adapter_to_model('default', lora_config,
                                   gradient_accumulation_steps=args.gradient_accumulation_steps)

    model.set_optimizer(optimizer_cls='AdamW', lr=args.lr)
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=5,
        num_training_steps=len(dataloader),
    )

    # Build experiment label
    backend_label = 'HyperParallel FSDP2' if args.backend == 'hp' else 'native FSDP2'
    tune_mode = 'LoRA' if args.use_lora else 'Full-tuning'
    exp_label = f'GA={args.gradient_accumulation_steps}, lr={args.lr}, {tune_mode}'
    logger.info(f'Backend: {backend_label} (Qwen3-0.6B, deterministic, {exp_label})')
    logger.info(get_device_placement())
    logger.info(model.get_train_configs())

    # Log trainable params
    unwrapped = model.strategy.unwrap_model(model.model)
    sample_params = [(n, p) for n, p in unwrapped.named_parameters() if p.requires_grad]
    total_params = sum(p.numel() for _, p in sample_params)
    if sample_params:
        _name, _p = sample_params[0]
        logger.info(f'{tune_mode}: {total_params:,} trainable params ({len(sample_params)} tensors), '
                     f'param dtype: {_p.dtype}')

    total_steps = len(dataloader) if args.max_steps is None else min(args.max_steps, len(dataloader))
    logger.info(f'Total steps: {total_steps}')
    log_memory('after model init')

    for step, batch in enumerate(dataloader):
        if args.max_steps is not None and step >= args.max_steps:
            break

        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()

        metric = model.calculate_metric(is_training=True)
        logger.info(f'Step {step}/{total_steps}, metric: {metric}')
        if step % 10 == 0:
            log_memory(f'step {step}')

    log_memory('final')
    logger.info(f'{backend_label} {tune_mode.lower()} complete. ({exp_label})')


if __name__ == '__main__':
    train()
