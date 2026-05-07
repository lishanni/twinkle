# Twinkle + HyperParallel GPU 硬件兼容性问题报告

> **版本**: 基于 main 分支 commit `d001934` (2026-05)
> **作者**: Twinkle 团队
> **适用范围**: 评估 twinkle + HyperParallel 在 NVIDIA GPU (CUDA) 上的可用性与迁移成本

---

## 1. 执行摘要

Twinkle 框架本身具备 **GPU/CUDA 和 NPU/Ascend 双平台抽象层**，但 HyperParallel 集成路径存在大量 NPU 硬编码和隐式依赖，导致 **GPU 环境下无法直接使用 HyperParallel FSDP2 后端**。具体表现为三类问题：

| 类别 | 严重程度 | 数量 | GPU 影响 |
|------|----------|------|----------|
| A. 硬编码 NPU 调用（运行时崩溃） | **Critical** | 6 处 | 直接报错 |
| B. NPU 专有优化/功能缺失（降级） | **High** | 8 处 | 功能或性能缺失 |
| C. Cookbook/示例仅支持 NPU | **Medium** | 10+ 处 | 无法直接复用 |

**结论**: Twinkle 原生 FSDP2 路径（`use_hyper_parallel=False`）可无修改运行于 GPU；HyperParallel 路径需要系统性改造才能在 GPU 上运行。

---

## 2. Twinkle 框架层兼容性分析

### 2.1 已有的设备抽象层

Twinkle 的平台抽象设计良好，通过 `Platform` 基类和设备检测逻辑实现了 GPU/NPU 双轨支持：

```
src/twinkle/utils/platforms/
├── base.py    # Platform 基类，统一接口
├── gpu.py     # GPU 实现：device_prefix='cuda', backend='nccl', CUDA_VISIBLE_DEVICES
├── npu.py     # NPU 实现：device_prefix='npu', backend='hccl', ASCEND_RT_VISIBLE_DEVICES
└── mps.py     # MPS 实现 (Apple Silicon)
```

**设备自动检测** (`base.py:23-38`):
```python
if shutil.which('npu-smi'):       # NPU 优先
    from .npu import NPU
elif shutil.which('nvidia-smi'):  # GPU 次之
    from .gpu import GPU
```

**框架工具函数** (`framework.py`): `empty_cache()`, `set_device()`, `synchronize()`, `seed_everything()` 等均通过 `Torch.is_gpu_available()` / `Torch.is_npu_available()` 条件分发，对 GPU 友好。

**Checkpoint 引擎** (`checkpoint_engine/mixin.py:16-27`):
```python
if Platform.get_platform().__name__ == 'GPU':
    from twinkle.checkpoint_engine import NCCLCheckpointEngine
elif Platform.get_platform().__name__ == 'NPU':
    from twinkle.checkpoint_engine import HCCLCheckpointEngine
```

### 2.2 DeviceMesh 默认值偏差

`DeviceMesh.device_type` 默认值为 `'cuda'` (`device_mesh.py:42`)，对 GPU 天然友好。但所有 NPU cookbook 显式传 `device_type='npu'`，需在迁移时修改。

---

## 3. HyperParallel 集成层兼容性问题

### 3.1 Critical: 运行时崩溃问题

#### 问题 A1: `log_npu_memory()` 硬编码 NPU API

**文件**: `src/twinkle/model/transformers/strategy/hyper_parallel.py:169-182`

```python
def log_npu_memory(self, model, logger, tag: str = '') -> None:
    if not (hasattr(torch, 'npu') and torch.npu.is_available()):
        return                                    # GPU 上静默跳过，不崩溃
    allocated = torch.npu.memory_allocated(...)    # 若强行调用会报错
    reserved = torch.npu.memory_reserved(...)
    max_allocated = torch.npu.max_memory_allocated(...)
```

**影响**: GPU 环境下该函数静默返回（不会崩溃），但会导致 **内存监控完全失效**。方法名亦暗示仅用于 NPU。

**修复建议**: 重命名为 `log_device_memory()`，通过 `Platform` 抽象统一调用 `torch.cuda` 或 `torch.npu`。

#### 问题 A2: Cookbook 中的 `torch.npu.device_count()` 硬编码

**涉及文件** (全部 cookbook 脚本):
- `compare_fsdp2_4b_deterministic.py:65`
- `compare_fsdp2_32b.py:48`
- `compare_fsdp2_30b_moe.py:48`
- `compare_fsdp2_fulltune_06b.py:49`
- `profile_fsdp2_compare_32b.py:27`
- `hyper_parallel_fsdp2_npu.py:23`
- `fsdp2_npu.py:23`

```python
world_size = torch.npu.device_count()  # GPU 上报 AttributeError
```

**影响**: GPU 环境下直接抛出 `AttributeError: module 'torch' has no attribute 'npu'`，脚本无法启动。

**修复建议**:
```python
world_size = (torch.npu.device_count() if hasattr(torch, 'npu') and torch.npu.is_available()
              else torch.cuda.device_count())
```

#### 问题 A3: Cookbook 中 `device_type='npu'` 硬编码

同上述文件，所有 DeviceMesh 创建均硬编码：
```python
device_mesh = DeviceMesh.from_sizes(fsdp_size=world_size, dp_size=1, device_type='npu')
```

**影响**: 在 GPU 环境下 `torch.distributed.DeviceMesh('npu', ...)` 会因找不到 `npu` 设备而报错。

**修复建议**: 使用 `Platform.device_prefix()` 动态获取。

#### 问题 A4: `torch_npu.profiler` 不可用

**文件**: `profile_fsdp2_compare_32b.py:13`, `profile_fsdp2_compare_30b_moe.py:13`

```python
import torch_npu.profiler as prof  # GPU 上 ModuleNotFoundError
```

**影响**: GPU 上无法 import，profiling 脚本完全不可用。

**修复建议**: 条件导入，GPU 上使用 `torch.profiler`。

#### 问题 A5: HyperParallel `symmetric_memory` 硬编码 NPU Stream

**文件**: `hyper-parallel-master/hyper_parallel/platform/torch/symmetric_memory/symmetric_memory.py:62-63`

```python
cls.comm_streams = [torch.npu.Stream() for _ in range(min(world_size, 16))]
cls.compute_streams = [torch.npu.Stream() for _ in range(world_size)]
```

**影响**: `torch.npu.Stream` 在 GPU 上不存在。当模型启用 symmetric memory 时会崩溃。

**修复建议**: 使用 `TorchPlatform.get_device_handle()` 获取正确的 Stream 类。

#### 问题 A6: 确定性种子设置中的 NPU 硬编码

**涉及文件**: `compare_fsdp2_4b_deterministic.py:35-36`, `compare_fsdp2_32b.py:24-25` 等

```python
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.manual_seed_all(42)
```

**影响**: 功能上无影响（`hasattr` 保护），但缺少对应的 `torch.cuda.manual_seed_all()` 调用。GPU 上确定性训练无法保证。

---

### 3.2 High: NPU 专有优化/功能缺失

#### 问题 B1: 通信后端 HCCL vs NCCL

**Twinkle 层**:
- `npu.py:117` → `device_backend()` 返回 `'hccl'`
- `gpu.py:21` → `device_backend()` 返回 `'nccl'`

**HyperParallel 层**: `fully_shard/api.py:476` 通过 `device_type` 自动选择：
```python
if device_type not in ("npu", "cuda"):
    raise AssertionError(...)
device_handle = platform.get_device_handle(device_type)
```

HyperParallel 使用 `getattr(torch, device_type)` 获取设备句柄，理论上 CUDA 可用。但实际运行时 **未在 GPU 上测试验证**。

#### 问题 B2: HCCL Socket 端口环境变量

**文件**: `src/twinkle/utils/platforms/npu.py:13-46`

```python
_HCCL_IF_BASE_PORT_ENV = 'HCCL_IF_BASE_PORT'
_HCCL_HOST_SOCKET_PORT_RANGE_ENV = 'HCCL_HOST_SOCKET_PORT_RANGE'
_HCCL_NPU_SOCKET_PORT_RANGE_ENV = 'HCCL_NPU_SOCKET_PORT_RANGE'
```

`ensure_hccl_socket_env()` 从 `master_port` 推导 HCCL 端口范围，避免多任务端口冲突。GPU 上的 NCCL 不需要此配置，但也不会产生干扰（仅在 NPU 平台分支调用）。

#### 问题 B3: HyperParallel 默认 `device_type='npu'`

**文件**: `hyper-parallel-master/hyper_parallel/platform/torch/platform.py:574`

```python
def get_device_handle(device_type: str = "npu"):
```

HyperParallel 的核心 API 默认使用 NPU。Twinkle 集成层通过 `hyper_parallel_config` 显式传递 `device_type`，但若遗漏则默认指向 NPU。

#### 问题 B4: `upcast_trainable_params` 行为差异

**文件**: `src/twinkle/model/transformers/strategy/hyper_parallel.py:32`

```python
upcast_trainable_params: bool = True  # 默认开启
```

此选项调用 HyperParallel 的 `model.to(torch.float32)` 将可训练参数转为 fp32。NPU 和 GPU 上此行为一致，但需注意 **HyperParallel 路径的参数精度管理方式与原生 FSDP2 不同**。

#### 问题 B5: `torch.npu` 全局对象检测优先级

**文件**: `src/twinkle/utils/framework.py:130-136`

```python
def get_current_device():
    if Torch.is_gpu_available():
        return torch.cuda.current_device()
    elif Torch.is_npu_available():
        return torch.npu.current_device()
```

在同时安装了 `torch_npu` 和 GPU 驱动的环境中，GPU 优先于 NPU。此逻辑正确，无需修改。

#### 问题 B6: `gather_object()` NPU 特殊路径

**文件**: `src/twinkle/utils/framework.py:46-60`

```python
if Platform.device_prefix() == 'npu':
    # NPU 上 HCCL 的 all_gather_object 在 8 卡时存在 hang 问题
    # 转用 Megatron 的 Gloo DP group
    if has_megatron:
        process_group = mpu.get_data_parallel_group_gloo(...)
```

GPU 上无需此 workaround（NCCL 的 `all_gather_object` 工作正常）。此分支仅在 NPU 平台触发，GPU 不受影响。

#### 问题 B7: Shell 脚本中的 NPU 检测逻辑

**涉及文件**:
- `cookbook/transformers/hyper_parallel_fsdp2_npu.sh`
- `cookbook/transformers/fsdp2_npu.sh`
- `cookbook/transformers/profile_fsdp2_compare_32b.sh`
- `cookbook/transformers/run_4b_ablation_serial.sh`

```bash
NPROC=$(npu-smi info | grep -c "910B")
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export HCCL_DETERMINISTIC=true
```

**影响**: Shell 脚本完全依赖 NPU 工具链，GPU 环境下无法使用。需提供对应的 `*_gpu.sh` 版本。

#### 问题 B8: Profiler API 差异

| 特性 | `torch_npu.profiler` (NPU) | `torch.profiler` (GPU) |
|------|---------------------------|----------------------|
| `ProfilerLevel.Level0/Level1` | 支持 | 无此概念 |
| `_ExperimentalConfig` | 支持 `profiler_level` | 不支持 |
| `ProfilerActivity.NPU` | 支持 | 无，使用 `CUDA` |
| TensorBoard trace handler | 支持 | 支持 |

**影响**: NPU profiling 脚本中的 `ProfilerLevel.Level1`、`ProfilerActivity.NPU` 等常量在 GPU 上不存在。

---

## 4. HyperParallel 库层兼容性分析

### 4.1 架构概览

HyperParallel 是华为昇腾团队开发的分布式训练加速库，核心功能包括：

```
hyper_parallel/
├── core/
│   ├── fully_shard/       # FSDP2 分片实现 (核心)
│   ├── dtensor/           # 分布式 Tensor
│   ├── shard/             # 张量分片
│   ├── activation_checkpoint/
│   ├── tensor_parallel/
│   ├── expert_parallel/
│   ├── context_parallel/
│   └── pipeline_parallel/
├── platform/
│   ├── torch/             # PyTorch 平台实现
│   └── mindspore/         # MindSpore 平台实现
├── integration/
│   └── llamafactory/      # LLaMA Factory 集成
└── scripts/               # 构建脚本
```

### 4.2 设备类型支持矩阵

**`core/dtensor/device_mesh.py:145-147`**:
```python
_VALID_DEVICE_TYPES = {
    PlatformType.PYTORCH: {"cpu", "cuda", "npu"},
    PlatformType.MINDSPORE: {"cpu", "gpu", "npu"},
}
```

**`core/fully_shard/api.py:475-479`**:
```python
device_type = mesh.device_type
if device_type not in ("npu", "cuda"):
    raise AssertionError(
        f"hyper_parallel.fully_shard support device in [torch.npu, torch.cuda], "
        f"but got '{device_type}'"
    )
```

**结论**: HyperParallel 在 API 层面声明支持 `cuda` 和 `npu`，但实际代码和测试 **几乎全部基于 NPU**。

### 4.3 NPU 硬编码的具体位置

| 文件 | 行号 | 硬编码内容 | GPU 替代 |
|------|------|-----------|----------|
| `platform/torch/platform.py` | 574 | `get_device_handle(device_type="npu")` | 默认值应动态检测 |
| `platform/torch/symmetric_memory/symmetric_memory.py` | 62-63 | `torch.npu.Stream()` | `torch.cuda.Stream()` |
| `integration/llamafactory/utils.py` | 48-51 | NPU 优先检测逻辑 | 已支持 CUDA fallback |
| `examples/torch/fully_shard/fsdp_demo.py` | 23-25 | `.npu()`, `torch.npu.set_device()` | 需 `.cuda()` |
| `tests/torch/common_net.py` | 24-44 | `.npu()` | 需 `.cuda()` |
| `tests/torch/utils.py` | 29-31 | `torch.npu.set_device()` | 需 `torch.cuda.set_device()` |

### 4.4 构建依赖

**`scripts/build_symmetric_memory.sh` / `build_multicore.sh`**:
```bash
# build_multicore.sh:19-20
# Requires: torch, torch_npu, and CANN vendor libs
```

对称内存和多核扩展的编译脚本依赖 `torch_npu` 和 CANN vendor 库。在 GPU 环境下，这些扩展 **无法编译**。

---

## 5. 环境变量差异对照

| 环境变量 | NPU | GPU | 说明 |
|----------|-----|-----|------|
| 设备可见性 | `ASCEND_RT_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` | Twinkle 已抽象 |
| 确定性计算 | `ASCEND_LAUNCH_BLOCKING=1`, `HCCL_DETERMINISTIC=true` | `CUDA_LAUNCH_BLOCKING=1`, `CUBLAS_WORKSPACE_CONFIG=:16:8` | Twinkle 已在 `seed_everything()` 中处理 |
| 通信后端 | `hccl` | `nccl` | Twinkle 已抽象 |
| Socket 端口 | `HCCL_IF_BASE_PORT`, `HCCL_HOST_SOCKET_PORT_RANGE`, `HCCL_NPU_SOCKET_PORT_RANGE` | 不需要 | NPU 专有，避免多任务端口冲突 |
| Profiler | `torch_npu.profiler` + `ProfilerLevel` | `torch.profiler` | API 不兼容 |

---

## 6. GPU 迁移方案

### 6.1 快速路径（仅需原生 FSDP2）

无需任何代码修改。设置 `use_hyper_parallel=False`（默认值），Twinkle 的原生 FSDP2 策略在 GPU 上完全可用：

```python
model = TransformersModel(
    model_id='Qwen/Qwen3-0.6B',
    device_mesh=device_mesh,
    # use_hyper_parallel 默认 False
)
```

### 6.2 HyperParallel GPU 迁移（中等工作量）

需改造的模块及优先级：

| 优先级 | 模块 | 工作量 | 说明 |
|--------|------|--------|------|
| P0 | Cookbook 脚本 | 小 | 替换 `torch.npu` → 动态检测，`device_type='npu'` → `'cuda'` |
| P0 | `symmetric_memory` Stream | 小 | `torch.npu.Stream()` → `torch.cuda.Stream()` |
| P1 | `log_npu_memory` | 小 | 重构为设备无关的 `log_device_memory()` |
| P1 | Profiler 脚本 | 中 | 条件导入 `torch_npu.profiler` / `torch.profiler` |
| P2 | Shell 启动脚本 | 小 | 新增 `*_gpu.sh` 版本 |
| P2 | HyperParallel 构建脚本 | 大 | symmetric_memory / multicore 扩展需要 CUDA 版本 |
| P3 | 测试用例 | 大 | 所有 `.npu()` 调用改为设备无关 |

### 6.3 推荐的设备抽象模式

```python
# 方案一：利用现有 Platform 层
from twinkle.utils.platforms import Platform

device_type = Platform.device_prefix()  # 'cuda' or 'npu'
world_size = Platform.get_local_world_size()

# 方案二：统一内存监控
def log_device_memory(logger, tag=''):
    import torch
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if hasattr(torch, 'npu') and torch.npu.is_available():
        alloc = torch.npu.memory_allocated(local_rank) / (1024**3)
        peak = torch.npu.max_memory_allocated(local_rank) / (1024**3)
    elif torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(local_rank) / (1024**3)
        peak = torch.cuda.max_memory_allocated(local_rank) / (1024**3)
    logger.info(f'[{tag}] device {local_rank} mem: alloc={alloc:.2f}GB, peak={peak:.2f}GB')

# 方案三：统一 profiler 导入
try:
    import torch_npu.profiler as prof
    PROF_ACTIVITY_DEVICE = prof.ProfilerActivity.NPU
except ImportError:
    import torch.profiler as prof
    PROF_ACTIVITY_DEVICE = prof.ProfilerActivity.CUDA
```

---

## 7. 已知限制

1. **HyperParallel 未在 GPU 上经过完整测试**: 虽然代码层面声明支持 `cuda`，但所有测试和示例均在 NPU 上运行，GPU 上可能存在未知 bug。

2. **Symmetric Memory 扩展不可用**: 该扩展的 C++ 编译依赖 CANN vendor 库，GPU 上无法构建。影响：`comm_fusion` 等通信优化功能受限。

3. **Expert Parallel 的 `npu_grouped_matmul`**: Expert Parallel 模块使用 `torch_npu.npu_grouped_matmul`，这是 NPU 专有算子，GPU 需使用 `torch.bmm` 或 `grouped_gemm` 替代。

4. **MindSpore 平台**: HyperParallel 的 MindSpore 后端仅支持 HCCL，无 GPU (NCCL) 支持。

5. **性能调优**: NPU 上的 prefetch、comm_fusion 等优化参数在 GPU 上可能不适用或效果不同，需要重新调优。

---

## 8. 总结

| 组件 | GPU 兼容性 | 说明 |
|------|-----------|------|
| Twinkle 框架核心 | 完全兼容 | Platform 抽象层完善 |
| Twinkle 原生 FSDP2 | 完全兼容 | 已有 GPU/NPU 双路径 |
| Twinkle + HyperParallel 集成 | **不兼容** | 6 处 Critical + 8 处 High |
| HyperParallel 库核心 (FSDP) | 声明兼容，未验证 | API 层支持 `cuda` |
| HyperParallel Symmetric Memory | **不兼容** | NPU Stream 硬编码 + CANN 编译依赖 |
| HyperParallel Expert Parallel | **不兼容** | `npu_grouped_matmul` NPU 专有 |
| Cookbook/示例脚本 | **不兼容** | 全面依赖 NPU API |

**最小可行路径**: 如仅需 GPU 上的 FSDP2 分布式训练，使用 Twinkle 原生 FSDP2 即可，零改动。若需在 GPU 上使用 HyperParallel 加速特性，建议按 6.2 节的优先级逐步改造。
