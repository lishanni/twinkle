import copy
import random

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_logger
from twinkle.data_format import Message, Trajectory
from twinkle.dataloader import DataLoader
from twinkle.dataset import DatasetMeta, LazyDataset
from twinkle.model import TransformersModel
from twinkle.preprocessor import Preprocessor

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DATASET_ID = 'ms://AI-ModelScope/LaTeX_OCR'
DATA_SLICE = range(32)
BATCH_SIZE = 4
SEED = 1234

logger = get_logger()


class LatexOCRProcessor(Preprocessor):

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        return Trajectory(
            messages=[
                Message(role='user', content='<image>Using LaTeX to perform OCR on the image.', images=[row['image']]),
                Message(role='assistant', content=row['text']),
            ]
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def _build_device_mesh(*, ulysses_size=None) -> DeviceMesh:
    return DeviceMesh.from_sizes(
        fsdp_size=2,
        ulysses_size=ulysses_size,
        device_type=Platform.get_platform().device_prefix(),
    )


def _create_dataloader(device_mesh: DeviceMesh) -> DataLoader:
    dataset = LazyDataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=DATA_SLICE))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=1024)
    dataset.map(LatexOCRProcessor)
    dataset.encode()
    return DataLoader(dataset=dataset, batch_size=BATCH_SIZE, device_mesh=device_mesh)


def _build_model(device_mesh: DeviceMesh) -> TransformersModel:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    model = TransformersModel(
        model_id=MODEL_ID,
        model_cls=Qwen3_5ForConditionalGeneration,
        device_mesh=device_mesh,
        strategy='native_fsdp',
    )
    model.model._no_split_modules = {'Qwen3_5DecoderLayer'}
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)
    return model


def _extract_grad_stats(model: TransformersModel, adapter_name: str = 'default'):
    grad_dict = {}
    grad_norm_sq = 0.0
    for name, param in model._get_trainable_parameters(adapter_name).items():
        grad = param.grad
        if grad is None:
            continue
        grad = grad.detach().float().cpu().clone()
        grad_dict[name] = grad
        grad_norm_sq += grad.pow(2).sum().item()
    return grad_dict, grad_norm_sq**0.5


def _compare_grad_dicts(base_grads, sp_grads):
    shared_names = sorted(set(base_grads) & set(sp_grads))
    if not shared_names:
        raise RuntimeError('No shared trainable gradients found between baseline and ulysses models.')

    worst_name = None
    worst_diff = 0.0
    for name in shared_names:
        base_grad = base_grads[name]
        sp_grad = sp_grads[name]
        if base_grad.shape != sp_grad.shape:
            raise RuntimeError(f'Gradient shape mismatch for {name}: {tuple(base_grad.shape)} vs {tuple(sp_grad.shape)}')
        diff = (base_grad - sp_grad).abs().max().item()
        if diff > worst_diff:
            worst_diff = diff
            worst_name = name
    return worst_name, worst_diff


def _reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _gather_objects(obj):
    if not dist.is_available() or not dist.is_initialized():
        return [obj]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def _run_single_step(model: TransformersModel, batch, *, seed: int):
    _seed_everything(seed)
    outputs = model.forward_backward(inputs=batch, adapter_name='default')
    loss = float(outputs['loss'])
    grad_dict, grad_norm = _extract_grad_stats(model, adapter_name='default')
    return loss, grad_dict, grad_norm


def compare():
    baseline_mesh = _build_device_mesh()
    ulysses_mesh = _build_device_mesh(ulysses_size=2)

    twinkle.initialize(
        mode='local',
        global_device_mesh=baseline_mesh,
        lazy_collect=False,
        seed=SEED,
    )

    dataloader = _create_dataloader(baseline_mesh)
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        raise RuntimeError('No batch available from dataloader.')

    _seed_everything(SEED)
    baseline_model = _build_model(baseline_mesh)
    _seed_everything(SEED)
    ulysses_model = _build_model(ulysses_mesh)
    ulysses_model.model.load_state_dict(copy.deepcopy(baseline_model.model.state_dict()), strict=True)

    baseline_loss, baseline_grads, baseline_grad_norm = _run_single_step(
        baseline_model, copy.deepcopy(batch), seed=SEED)
    ulysses_loss, ulysses_grads, ulysses_grad_norm = _run_single_step(
        ulysses_model, copy.deepcopy(batch), seed=SEED)

    loss_diff = abs(baseline_loss - ulysses_loss)
    grad_norm_diff = abs(baseline_grad_norm - ulysses_grad_norm)
    worst_name, worst_grad_diff = _compare_grad_dicts(baseline_grads, ulysses_grads)

    device = next(iter(ulysses_model._get_trainable_parameters('default').values())).device
    max_loss_diff = _reduce_max(loss_diff, device)
    max_grad_norm_diff = _reduce_max(grad_norm_diff, device)
    max_grad_diff = _reduce_max(worst_grad_diff, device)

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    local_report = {
        'rank': rank,
        'baseline_loss': baseline_loss,
        'ulysses_loss': ulysses_loss,
        'loss_diff': loss_diff,
        'baseline_grad_norm': baseline_grad_norm,
        'ulysses_grad_norm': ulysses_grad_norm,
        'grad_norm_diff': grad_norm_diff,
        'worst_grad_name': worst_name,
        'worst_grad_diff': worst_grad_diff,
    }
    reports = _gather_objects(local_report)

    if rank == 0:
        logger.info(f'Max loss diff across ranks: {max_loss_diff:.8f}')
        logger.info(f'Max grad norm diff across ranks: {max_grad_norm_diff:.8f}')
        logger.info(f'Max per-parameter grad diff across ranks: {max_grad_diff:.8f}')
        for report in reports:
            logger.info(
                'Rank %(rank)s | baseline_loss=%(baseline_loss).8f | ulysses_loss=%(ulysses_loss).8f | '
                'loss_diff=%(loss_diff).8f | baseline_grad_norm=%(baseline_grad_norm).8f | '
                'ulysses_grad_norm=%(ulysses_grad_norm).8f | grad_norm_diff=%(grad_norm_diff).8f | '
                'worst_grad=%(worst_grad_name)s | worst_grad_diff=%(worst_grad_diff).8f' % report)


if __name__ == '__main__':
    compare()
