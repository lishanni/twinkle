"""HyperParallel FSDP2 on NPU — comm_fusion + prefetch enabled, aligned with native baseline."""
import os

from peft import LoraConfig
from tqdm import tqdm

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

# Auto-detect world_size from available NPU count
world_size = torch.npu.device_count()
device_mesh = DeviceMesh.from_sizes(fsdp_size=world_size, dp_size=1, device_type='npu')
twinkle.initialize(mode='local', nproc_per_node=world_size, global_device_mesh=device_mesh)

logger = get_logger()

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DATASET_ID = 'ms://swift/self-cognition'


def log_memory(tag: str = '') -> None:
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if hasattr(torch, 'npu') and torch.npu.is_available():
        allocated = torch.npu.memory_allocated(local_rank) / (1024 ** 3)
        reserved = torch.npu.memory_reserved(local_rank) / (1024 ** 3)
        peak = torch.npu.max_memory_allocated(local_rank) / (1024 ** 3)
        prefix = f'[{tag}] ' if tag else ''
        logger.info(f'{prefix}NPU {local_rank} mem: alloc={allocated:.2f}GB, reserved={reserved:.2f}GB, peak={peak:.2f}GB')


def eval_model(model):
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(100)))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    for _, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    return model.calculate_metric(is_training=False)


def train():
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(1000)))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=8)

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
            'comm_fusion': True,
            'enable_prefetch': True,
            'num_forward_prefetch': 1,
            'num_backward_prefetch': 1,
            'upcast_trainable_params': True,
        },
    )
    model.model._no_split_modules = {'Qwen3_5DecoderLayer'}

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4)
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=5,
        num_training_steps=len(dataloader),
    )

    logger.info(f'Backend: hyper_parallel FSDP2 (comm_fusion=True, prefetch=True, upcast=True)')
    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    log_memory('after model init')

    best_loss = 99.0
    for step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        metric = model.calculate_metric(is_training=True)
        logger.info(f'Step {step}/{len(dataloader)}, metric: {metric}')
        if step % 10 == 0:
            log_memory(f'step {step}')
        if step > 0 and step % 40 == 0:
            metrics = eval_model(model)
            logger.info(f'Eval metric: {metrics}')
            if best_loss > float(metrics['loss']):
                model.save(f'hp-checkpoint-{step}')
                best_loss = float(metrics['loss'])
    model.save('hp-last-checkpoint')
    log_memory('final')


if __name__ == '__main__':
    train()
