# Twinkle + HyperParallel GPU 硬件兼容性问题报告

> **版本**: 基于 main 分支 commit `9ac2014` (2026-05)
> **作者**: HyperParallel lishanni
> **适用范围**: 评估 twinkle + HyperParallel 在 NVIDIA GPU (CUDA) 上的可用性与迁移成本

---

## 1. 执行摘要

Twinkle 框架具备完善的 **GPU/CUDA 和 NPU/Ascend 双平台抽象层**（`Platform` 基类、`Torch` 工具函数、`DeviceMesh` 等）。HyperParallel 集成严格遵循 Twinkle 的策略模式设计，`HyperParallelStrategy` 继承自 `AccelerateStrategy`，与 `NativeFSDPStrategy` 共享统一接口，**无需在 `TransformersModel` 中引入任何 HyperParallel 特有逻辑**。

集成层的 Cookbook 脚本、设备检测、内存监控、确定性种子等均已通过 `Platform` / `Torch` 抽象实现设备无关化，**HyperParallel FSDP2 后端在 GPU 上可开箱即用**。

当前剩余的 GPU 兼容性问题仅限于外围工具层面：

| 层面 | 严重程度 | 说明 |
|------|----------|------|
| Profiling 脚本内部常量 | **Medium** | `ProfilerActivity.NPU` 等常量需条件化（顶层导入已处理） |
| Shell 启动脚本 | **Medium** | NPU 专用环境变量，需新增 GPU 分支 |

---

## 2. 已完成的设备无关化改造

### 2.1 Twinkle 框架层

Twinkle 通过 `Platform` 基类实现 GPU/NPU 双轨支持：

```
src/twinkle/utils/platforms/
├── base.py    # Platform 基类，统一接口
├── gpu.py     # GPU 实现：device_prefix='cuda', backend='nccl', CUDA_VISIBLE_DEVICES
├── npu.py     # NPU 实现：device_prefix='npu', backend='hccl', ASCEND_RT_VISIBLE_DEVICES
└── mps.py     # MPS 实现 (Apple Silicon)
```

**Checkpoint 引擎**、**框架工具函数**（`empty_cache()`, `set_device()`, `synchronize()`, `seed_everything()`）均已通过 `Platform` 抽象实现双平台支持。

### 2.2 HyperParallel 策略集成层

重构后的策略架构统一继承自 `AccelerateStrategy`，三个并行后端共享一致接口：

```
AccelerateStrategy          # 基类，定义 wrap_optimizer / adjust_optimizer_kwargs / log_device_memory 等
├── HyperParallelStrategy   # HyperParallel FSDP2，override 接口方法
├── NativeFSDPStrategy      # 原生 FSDP2，实现 EP 相关逻辑
└── (默认 Accelerate 路径)   # accelerate 分布式策略
```

**已完成的改造项**：

| 改造项 | 说明 |
|--------|------|
| `log_npu_memory` → `log_device_memory` | 通过 `Torch.is_npu_available()` / `Torch.is_gpu_available()` 双路径支持 NPU/GPU 内存监控 |
| Cookbook 设备无关化 | 8 个 cookbook 脚本全部改用 `Platform.get_local_world_size()` 和 `Platform.device_prefix()` |
| HP config `device_type` | 由硬编码 `'npu'` 改为动态变量 `device_type = Platform.device_prefix()` |
| 确定性种子双平台 | 新增 `elif torch.cuda.is_available()` 分支，GPU 确定性训练可用 |
| 策略接口清理 | `transformers.py` 中 `hasattr` 检查和 HP 直接 import 全部替换为统一策略接口调用 |
| Profiler 条件导入 | `try: import torch_npu.profiler` / `except: import torch.profiler` |

### 2.3 GPU 使用示例

```python
from twinkle.utils.platforms import Platform
from twinkle import DeviceMesh, twinkle

device_type = Platform.device_prefix()  # GPU 环境自动返回 'cuda'
world_size = Platform.get_local_world_size()

device_mesh = DeviceMesh.from_sizes(fsdp_size=world_size, dp_size=1, device_type=device_type)
twinkle.initialize(mode='local', nproc_per_node=world_size, global_device_mesh=device_mesh)

model = TransformersModel(
    model_id='Qwen/Qwen3-0.6B',
    device_mesh=device_mesh,
    use_hyper_parallel=True,
    hyper_parallel_config={
        'tp_size': 1,
        'device_type': device_type,
        'param_dtype': 'bf16',
        'reduce_dtype': 'bf16',
    },
)
```

---

## 3. 剩余兼容性问题

### 3.1 Profiler 脚本内部常量硬编码

**严重程度**: Medium
**涉及文件**: `profile_fsdp2_compare_32b.py`, `profile_fsdp2_compare_30b_moe.py`

顶层 `try/except` 条件导入已实现，但脚本内部仍使用 NPU 专有常量：

```python
# 当前代码（line 133-135）
experimental_config = prof._ExperimentalConfig(profiler_level=prof.ProfilerLevel.Level1)
prof.profile(
    activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],  # GPU 上不存在
)
```

**影响**: GPU 上 `torch.profiler` 没有 `ProfilerLevel`、`ProfilerActivity.NPU`、`_ExperimentalConfig` 等 API，Profiling 功能不可用。

**修复方案**: 条件化 Profiler 活动常量和实验配置：

```python
PROF_ACTIVITY_DEVICE = getattr(prof.ProfilerActivity, 'NPU', None) or prof.ProfilerActivity.CUDA
experimental_config = None
if hasattr(prof, 'ProfilerLevel'):
    experimental_config = prof._ExperimentalConfig(profiler_level=prof.ProfilerLevel.Level1)
```

### 3.2 Shell 启动脚本

**严重程度**: Medium
**涉及文件**: `hyper_parallel_fsdp2_npu.sh`, `fsdp2_npu.sh`, `profile_fsdp2_compare_32b.sh`

Shell 脚本中硬编码 NPU 环境变量和工具链：

```bash
NPROC=$(npu-smi info | grep -c "910B")
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export HCCL_DETERMINISTIC=true
```

**影响**: GPU 环境下 Shell 脚本不可用，需手动设置 `CUDA_VISIBLE_DEVICES` 等环境变量。

**修复方案**: 新增设备自动检测逻辑，或提供 `*_gpu.sh` 版本：

```bash
if command -v nvidia-smi &>/dev/null; then
    NPROC=$(nvidia-smi -L | wc -l)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
else
    NPROC=$(npu-smi info | grep -c "910B")
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
fi
```

---

## 4. HyperParallel 库层 GPU 兼容性

### 4.1 设备类型支持

HyperParallel 在 API 层面声明支持 `cuda` 和 `npu`：

```python
# core/fully_shard/api.py
if device_type not in ("npu", "cuda"):
    raise AssertionError(...)

# core/dtensor/device_mesh.py
_VALID_DEVICE_TYPES = {PlatformType.PYTORCH: {"cpu", "cuda", "npu"}}
```

Twinkle 集成层通过 `hyper_parallel_config` 显式传递 `device_type`，可正确路由到 CUDA 后端。

### 4.2 Symmetric Memory 扩展

HyperParallel 的 `symmetric_memory` 模块基于 NPU 硬件单边通信能力（CANN vendor 库），属于 NPU 平台专有的底层通信优化特性，GPU 上不适用。GPU 环境下关闭 `comm_fusion` 和 `enable_prefetch` 即可，FSDP2 核心分片功能不受影响。

### 4.3 Expert Parallel NPU 专有算子

Expert Parallel 模块使用 `torch_npu.npu_grouped_matmul`（NPU 专有算子），GPU 需使用 `torch.bmm` 或 `grouped_gemm` 替代。当前 Twinkle 集成层已在 `_decide_strategy` 中禁止 `use_hyper_parallel=True` 与 `expert_parallel` 同时启用，避免运行时错误。

### 4.4 GPU 验证状态

HyperParallel FSDP2 核心功能已在 NVIDIA A100 多卡环境下验证通过，bf16 混合精度、LoRA 微调、forward/backward + optimizer step 全流程正常，GPU 上 FSDP2 能力 Ready。

---

## 5. 环境变量差异对照

| 环境变量 | NPU | GPU | 说明 |
|----------|-----|-----|------|
| 设备可见性 | `ASCEND_RT_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` | Twinkle 已抽象 |
| 确定性计算 | `ASCEND_LAUNCH_BLOCKING=1`, `HCCL_DETERMINISTIC=true` | `CUDA_LAUNCH_BLOCKING=1`, `CUBLAS_WORKSPACE_CONFIG=:16:8` | Cookbook 已双平台支持 |
| 通信后端 | `hccl` | `nccl` | Twinkle 已抽象 |
| Socket 端口 | `HCCL_IF_BASE_PORT` 等 | 不需要 | NPU 专有，GPU 不受影响 |
| Profiler | `torch_npu.profiler` + `ProfilerLevel` | `torch.profiler` | 需条件化常量（见 3.1） |

---

## 6. GPU 迁移方案

### 6.1 推荐路径：使用 HyperParallel FSDP2 后端

Twinkle + HyperParallel 的 GPU 使用路径与 NPU 基本一致，核心区别仅在于设备类型由 `Platform` 自动检测：

```python
from twinkle.utils.platforms import Platform
from twinkle.utils.framework import Torch

device_type = Platform.device_prefix()  # GPU → 'cuda', NPU → 'npu'
world_size = Platform.get_local_world_size()
```

**HyperParallel 配置要点**：

```python
hp_config = {
    'tp_size': 1,
    'device_type': device_type,           # 动态获取，勿硬编码
    'param_dtype': 'bf16',
    'reduce_dtype': 'bf16',
}
```

### 6.2 需处理的遗留项

| 优先级 | 项目 | 工作量 | 说明 |
|--------|------|--------|------|
| P1 | Profiler 常量条件化 | 小 | `ProfilerActivity.NPU` → 动态选择 |
| P1 | Shell 脚本设备检测 | 小 | 新增 GPU 自动检测分支 |

### 6.3 设备无关代码模式

```python
# 确定性种子（Cookbook 已采用此模式）
torch.manual_seed(42)
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.manual_seed_all(42)
elif torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 内存监控（已实现在 HyperParallelStrategy.log_device_memory）
from twinkle.utils.framework import Torch
if Torch.is_npu_available():
    allocated = torch.npu.memory_allocated(local_rank)
elif Torch.is_gpu_available():
    allocated = torch.cuda.memory_allocated(local_rank)
```

---

## 7. 总结

| 组件 | GPU 兼容性 | 说明 |
|------|-----------|------|
| Twinkle 框架核心 | 完全兼容 | Platform 抽象层完善 |
| Twinkle + HyperParallel 集成层 | **完全兼容** | 策略接口已统一、Cookbook 已设备无关化 |
| HyperParallel FSDP2 核心 | **已验证通过** | A100 多卡 FSDP2 训练全流程正常 |
| HyperParallel Symmetric Memory | NPU 专有 | 基于 NPU 单边通信能力，GPU 不适用，不影响核心功能 |
| Profiler 脚本 | 部分兼容 | 顶层导入已条件化，内部常量待处理 |
| Shell 启动脚本 | 不兼容 | NPU 专用，需新增 GPU 分支 |

**结论**: 经过策略接口重构和 Cookbook 设备无关化改造，Twinkle 集成层的 HyperParallel GPU 兼容性问题已基本解决。HyperParallel FSDP2 核心功能已在 A100 多卡环境验证通过，GPU 用户可通过 `use_hyper_parallel=True` 直接启用。
