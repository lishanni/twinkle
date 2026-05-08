# Twinkle + HyperParallel Python 三方组件依赖分析报告

> **环境**: conda env `hyper_twinkle` on openEuler 22.03 LTS-SP1 (aarch64)
> **Python**: 3.11.15 | **硬件**: Ascend 910B3 × 8
> **验证时间**: 2026-05-07 | **基于 commit**: `9ac2014`

---

## 1. 执行摘要

当前 `hyper_twinkle` conda 环境共安装 **169 个包**。核心依赖链为：

```
twinkle-kit 0.3.0.dev0
├── numpy >=2.0.0,<2.3.0              ← HP 实际兼容 2.x，后续版本放宽声明├── datasets, omegaconf, fastapi
├── modelscope[framework] >=1.34.0
├── safetensors, peft >=0.11.0, transformers
├── [transformers extra] accelerate, torch >=2.6.0, torchvision
└── [hyper_parallel extra] hyper_parallel

hyper_parallel 0.1.0
├── numpy >=1.20.0,<2.0.0             ← 声明 (实际兼容 2.x，后续放宽)
└── [隐式] torch, torch_npu, CANN hccl
```

**关键事项**:

| # | 事项 | 说明 |
|---|------|------|
| 1 | numpy 版本声明差异 | HP 声明 `<2.0.0` 但实际兼容 numpy 2.x，后续版本将放宽 |
| 2 | torch_npu 双重注册 | 需 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，已在 Cookbook 和测试中统一配置 |
| 3 | HyperParallel 声明依赖不完整 | 仅声明 numpy，隐式依赖 torch/torch_npu，后续版本补全 |

---

## 2. 核心依赖版本矩阵

### 2.1 直接依赖（声明 + 实际安装）

| 包名 | Twinkle 声明范围 | HP 声明范围 | 实际版本 | 满足 Twinkle | 满足 HP |
|------|-----------------|------------|---------|-------------|---------|
| numpy | ≥2.0.0,<2.3.0 | ≥1.20.0,<2.0.0 | **2.2.6** | YES | YES * |
| torch | ≥2.6.0,<3.0.0 | (隐式) | **2.7.1+cpu** | YES | YES |
| transformers | any | (隐式) | **5.6.2** | YES | YES |
| accelerate | any | (隐式) | **1.11.0** | YES | YES |
| peft | ≥0.11.0,≤0.19.0 | (隐式) | **0.18.1** | YES | YES |
| datasets | any | (无) | **4.8.4** | YES | — |
| modelscope | [framework]≥1.34.0 | (无) | **1.35.3** | YES | — |
| omegaconf | ≥2.3.0,<3.0.0 | (无) | **2.3.0** | YES | — |
| safetensors | any | (隐式) | **0.7.0** | YES | YES |
| fastapi | any | (无) | **0.135.3** | YES | — |

> \* numpy 2.2.6 虽超出 HP 声明范围，但实际运行中 Qwen3.5-4B/30B/32B 等多个网络 LoRA 训练均正常通过，证明 HP 核心功能与 numpy 2.x 兼容。HP 后续版本将放宽声明上限。

### 2.2 平台专有依赖

| 包名 | 版本 | 来源 | GPU 可用性 |
|------|------|------|-----------|
| torch_npu | 2.7.1 | PyPI | 仅 NPU |
| hccl | 0.1.0 | CANN 8.5.0 | 仅 NPU |
| CANN toolkit | 8.5.0 | 系统安装 (/home/lsn/Ascend/) | 仅 NPU |

### 2.3 传递依赖（关键路径）

```
twinkle-kit
├── modelscope[framework] → requests, setuptools, urllib3
├── peft → accelerate, transformers, torch, huggingface_hub
│   └── transformers → tokenizers, regex, tqdm, typer, safetensors, huggingface_hub
│       └── huggingface_hub → hf_transfer, hf_xet
├── accelerate → torch, numpy, packaging, pyyaml, psutil
├── datasets → pandas, pyarrow, dill, multiprocess, fsspec, httpx
│   └── pandas → python-dateutil, pytz
│   └── httpx → httpcore, h11, anyio
└── hyper_parallel → numpy (only declared)
    └── [implicit] torch, torch_npu, transformers (via llamafactory integration)

torch_npu
└── torch (declared)
    └── filelock, fsspec, jinja2, networkx, sympy, typing-extensions
```

---

## 3. 实际验证结果

### 3.1 Import 可用性测试

以下测试使用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 环境变量（**必须**，否则 torch_npu 双重注册崩溃）：

| 模块 | 导入结果 | 版本 | 备注 |
|------|---------|------|------|
| `twinkle` | OK | 0.3.0.dev0 | 核心导入正常 |
| `twinkle.DeviceMesh` | OK | — | |
| `torch` | OK | 2.7.1+cpu | **CPU 构建**，无 CUDA 支持 |
| `torch.npu` | OK | — | 8 卡 NPU 可用 |
| `torch.cuda` | FAIL | — | CPU 构建无 CUDA |
| `torch_npu` | OK | 2.7.1 | 需先设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |
| `transformers` | OK | 5.6.2 | 需先导入 torch_npu |
| `peft` | OK | 0.18.1 | 需先导入 torch_npu |
| `accelerate` | OK | 1.11.0 | 需先导入 torch_npu |
| `hyper_parallel` | OK | — | |
| `hyper_parallel.core.fully_shard.api` | OK | — | fully_shard, get_model_state_dict |
| `hyper_parallel.SkipDTensorDispatch` | OK | — | |
| `hyper_parallel.integration.llamafactory.utils` | OK | — | fsdp2_prepare_model |
| `safetensors` | OK | 0.7.0 | |
| `datasets` | OK | 4.8.4 | |
| `omegaconf` | OK | 2.3.0 | |
| `fastapi` | OK | 0.135.3 | |
| `hccl` | OK | 0.1.0 | 来自 CANN 8.5.0 |

### 3.2 关键验证：Import 顺序敏感

```python
# 错误顺序 — 会崩溃:
from peft import LoraConfig          # peft → transformers → is_torch_npu_available()
                                     # → import torch_npu → RuntimeError!

# 正确顺序 — 需环境变量:
#   export TORCH_DEVICE_BACKEND_AUTOLOAD=0
import torch
import torch_npu                     # 必须在 transformers/peft 之前
from peft import LoraConfig          # OK

# Cookbook 实际顺序 (from peft 在 import torch 之前):
from peft import LoraConfig          # peft 内部会触发 import torch
import torch                         # torch 已在 sys.modules 中
```

** Cookbook 脚本能正常工作的原因**: torchrun 执行环境下，torch 的 auto-load 行为与直接 `python -c` 不同；且 CANN 环境可能已设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。

---

## 4. numpy 版本声明差异

HyperParallel `requirements.txt` 声明 `numpy>=1.20.0,<2.0.0`，Twinkle 声明 `numpy>=2.0.0,<2.3.0`，二者在声明层面无交集。

**实际验证**: 在 `numpy=2.2.6` 环境下，已通过以下网络的 LoRA 训练验证，无任何 numpy 相关运行时错误：

| 模型 | 规模 | 训练步数 | 结果 |
|------|------|---------|------|
| Qwen3.5-4B | 4B | 100+ steps | 正常 |
| Qwen3.5-32B | 32B | 50+ steps | 正常 |
| Qwen3-30B-MoE | 30B MoE | 50+ steps | 正常 |
| Qwen3.5-0.6B | 0.6B | 完整 epoch | 正常 |

**结论**: HyperParallel 核心功能实际兼容 numpy 2.x。声明范围将在后续版本中放宽为 `numpy>=1.20.0`（去掉 `<2.0.0` 上限），待完成全量回归验证后生效。

---

## 5. torch/torch_npu 交互问题

### 5.1 问题机制

PyTorch 2.7+ 引入了设备后端自动加载机制 (`TORCH_DEVICE_BACKEND_AUTOLOAD`)：

```
Python 启动
  │
  ├── TORCH_DEVICE_BACKEND_AUTOLOAD=1 (默认)
  │   └── import torch → 自动发现 torch_npu → 注册 torch.npu 后端
  │       └── 后续 import torch_npu → 检测到已注册的 npu → RuntimeError
  │           "Two accelerators: npu and npu"
  │
  └── TORCH_DEVICE_BACKEND_AUTOLOAD=0
      └── import torch → 不自动加载后端
          └── import torch_npu → 手动注册 → 正常
```

### 5.2 影响范围

- `transformers 5.6.2` 的 `is_torch_npu_available()` 在模块级调用 `import torch_npu`
- `modelscope 1.35.3` 的扩展加载机制也会触发此问题
- 所有依赖 transformers 的包 (peft, accelerate, trl) 间接受影响

### 5.3 解决方案

在运行任何 Twinkle/HP 代码前设置环境变量：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

或在 Python 代码最开头（任何其他 import 之前）：

```python
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch
import torch_npu  # 必须显式导入
```

---

## 6. Twinkle 可选依赖

Twinkle 的 `pyproject.toml` 声明了多个可选依赖组：

| Extra | 声明内容 | 安装状态 | 说明 |
|-------|---------|---------|------|
| `transformers` | accelerate, torch≥2.6, torchvision | **已安装** | 核心训练依赖 |
| `hyper_parallel` | hyper_parallel | **已安装** | HP 加速后端 |
| `kernels` | kernels | **未安装** | 自定义算子 |
| `megatron` | megatron-core≥0.12, transformer-engine, mcore_bridge | **未安装** | Megatron 分布式训练 |
| `vllm` | vllm≥0.11 | **未安装** | vLLM 推理引擎 |
| `ray` | ray[serve] | **未安装** | Ray 分布式 |
| `tinker` | tinker==0.14.0 | **未安装** | Tinker 可视化 |
| `docs` | sphinx, myst_parser, etc. | **未安装** | 文档构建 |

---

## 7. 完整依赖关系图

### 7.1 Twinkle 直接 + 传递依赖 (已安装)

```
twinkle-kit 0.3.0.dev0
├── numpy 2.2.6
├── datasets 4.8.4
│   ├── dill 0.3.8
│   ├── filelock 3.25.2
│   ├── fsspec 2025.3.0
│   ├── httpx 0.28.1 → httpcore 1.0.9 → h11 0.16.0
│   ├── huggingface_hub 1.11.0
│   ├── multiprocess 0.70.16
│   ├── pandas 2.3.3 → python-dateutil, pytz
│   ├── pyarrow 24.0.0
│   ├── requests 2.33.1 → urllib3, certifi, charset-normalizer, idna
│   └── xxhash 3.6.0
├── omegaconf 2.3.0
├── fastapi 0.135.3
│   ├── starlette 0.52.1 → anyio
│   ├── pydantic 2.12.3 → pydantic_core
│   └── uvicorn 0.44.0
├── modelscope[framework] 1.35.3
│   ├── filelock, packaging, requests, setuptools, tqdm, urllib3
│   └── [隐式] torch, torch_npu, transformers (运行时)
├── safetensors 0.7.0
├── peft 0.18.1
│   ├── accelerate 1.11.0 → torch, numpy, psutil, pyyaml
│   ├── transformers 5.6.2 → tokenizers, regex, typer, huggingface_hub
│   └── torch 2.7.1+cpu
├── torch 2.7.1+cpu
│   ├── filelock, fsspec, jinja2, networkx, sympy, typing-extensions
│   └── [auto-loads] torch_npu 2.7.1 (if TORCH_DEVICE_BACKEND_AUTOLOAD=1)
└── hyper_parallel 0.1.0 (editable: /home/lsn/hyper-parallel-master)
    ├── numpy (声明: >=1.20.0,<2.0.0, 实际兼容 2.x)
    └── [隐式] torch, torch_npu, transformers
```

---

## 8. GPU 环境依赖

若需在 NVIDIA GPU 环境运行 Twinkle + HyperParallel，依赖差异如下：

| 类别 | NPU 环境 (当前) | GPU 环境 (目标) |
|------|----------------|----------------|
| torch | 2.7.1+cpu | 需要 2.7.1+cu** (CUDA 构建) |
| torch_npu | 2.7.1 | **不需要** |
| hccl | 0.1.0 (CANN) | **不需要** |
| nvidia-nccl | 已安装但不可用 | **需要** |
| nvidia-cublas/cudnn | 已安装但不可用 | **需要** |
| CANN toolkit | 8.5.0 | **不需要** |
| triton | 3.6.0 | 需要 (torch CUDA 版自带) |
| TORCH_DEVICE_BACKEND_AUTOLOAD | 需设为 0 | 不需要 (无 torch_npu) |
| numpy 声明差异 | 存在 | 相同 |

---

## 9. 总结与建议

### 必须解决

1. **numpy 版本声明差异**: HP 声明 `<2.0.0` 但实际兼容 numpy 2.x（已通过多网络验证）。HP 后续版本将放宽声明上限。
2. **torch_npu 双重注册**: 需在脚本入口设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`（已在 Cookbook 和测试中统一配置）。

### 建议改进

3. **HyperParallel 声明依赖不完整**: `requirements.txt` 仅声明 numpy，应添加 `torch` 和 `torch_npu`（条件）。
4. **清理无用 NVIDIA 包**: 当前 aarch64 NPU 环境中的 15 个 nvidia-* 包和 cuda-* 包为死重。
5. **Import 顺序文档化**: 在 Twinkle 文档中说明 `import torch; import torch_npu` 必须在 `from peft import ...` 和 `from transformers import ...` 之前。

---

## 10. Twinkle Kernel 优化算子与 HyperParallel 兼容性分析

### 10.1 Twinkle Kernel 模块架构

```
src/twinkle/kernel/
├── __init__.py      # 入口: kernelize_model(), register_kernels()
├── base.py          # 设备/模式定义, 环境变量控制
├── registry.py      # LayerRegistry, FunctionRegistry, ExternalLayerRegistry
├── layer.py         # Layer kernel 注册/应用 (替换 Module.forward)
├── function.py      # Function kernel 注册/应用 (monkey-patch 模块函数)
└── csrc/
    └── placeholder  # 空目录, 无原生 C++ 算子
```

**核心机制**:

| 类型 | 机制 | 替换对象 | 注册 API |
|------|------|---------|---------|
| Layer Kernel | 替换 `nn.Module.forward()` | 模型层（如 `LlamaAttention`） | `register_layer_kernel()` |
| Function Kernel | `setattr(module, func_name, impl)` | 模块级函数（如 `torch.nn.functional.silu`） | `register_function_kernel()` |
| External Layer | `replace_kernel_forward_from_hub()` | 任意 PyTorch 类 | `register_external_layer()` |

**关键特性**:
- 实际算子实现来自 HuggingFace Hub 的 `kernels` 包（预编译优化 kernel）
- 通过 `TWINKLE_USE_KERNELS` 环境变量全局开关（默认 YES）
- 支持设备过滤 (`cuda`/`npu`/`mps`/`cpu`) 和模式过滤 (`train`/`inference`/`compile`)
- 有完整的 fallback 机制：kernel 不可用时回退到原始实现
- **当前环境未安装 `kernels` 包** → kernel 模块所有注册/应用操作均为空操作 (no-op)

### 10.2 HyperParallel 的算子调度机制

HyperParallel 通过 DTensor dispatch 系统拦截所有张量操作：

```
用户调用 torch.matmul(dtensor_a, dtensor_b)
  │
  ├── OpDispatcher.dispatch()  (_op_dispatch.py:998)
  │   ├── _should_bypass_dispatch? → SkipDTensorDispatch 上下文
  │   ├── random ops → _dispatch_random_op
  │   └── layout inference → _dispatch_layout_infer
  │       ├── 查找 _DISTRIBUTED_OPS 注册表
  │       ├── infer_layout() → 推理输出分片布局
  │       ├── get_expand_impl() → 获取分片执行实现
  │       └── op_impl(*local_args) → 在 local shard 上执行
  │
  └── 输出包装为 DTensor
```

**HP 提供的算子扩展点**:

| 扩展点 | 文件 | 用途 |
|--------|------|------|
| `register_distributed_op()` | `core/shard/ops/parallel_ops_register.py:21` | 注册分片感知的分布式算子 |
| `SkipDTensorDispatch` | `core/dtensor/dtensor.py:35` | 上下文管理器，绕过 DTensor dispatch |
| `add_no_skip_ops()` | `core/shard/_op_dispatch.py` | 指定必须走 dispatch 的算子白名单 |

HP 内置 **50+ 分布式算子**（matmul, embedding, attention 等），通过 `parallel_ops.py` 中的 `DistributedOp` 基类自动注册。

### 10.3 兼容性矩阵：Twinkle Kernel × HyperParallel

#### Layer Kernel（替换 Module.forward）

**结论：天然兼容，无需额外适配。**

```
执行路径:
  model.forward(input)       ← Twinkle kernel 替换了 forward()
    → kernel_forward(input)
      → 内部调用 torch.matmul, torch.nn.functional.linear 等标准 ops
      → HP DTensor dispatch 自动拦截这些标准 ops → 分片计算
      → 输出
```

原因：HP 的 `fully_shard` 在**参数级别**分片（将 `nn.Parameter` 转为 DTensor shard），不修改 `forward()` 逻辑。Layer kernel 替换的是 `forward()` 方法，两者作用在不同层面：

| 层面 | HP fully_shard | Twinkle Layer Kernel |
|------|---------------|---------------------|
| 作用对象 | `nn.Parameter` → DTensor shard | `Module.forward()` → 优化实现 |
| 修改方式 | 参数替换 | 方法替换 |
| 交互点 | forward() 内部调用的 torch ops | — |

**使用示例**:

```python
from twinkle.kernel import register_layer_kernel, kernelize_model
from twinkle.model.transformers import TransformersModel

# 1. 注册 kernel
register_layer_kernel(
    'LlamaAttention',
    repo_id='kernels-community/llama-attention',
    device='npu',
    mode='train',
)

# 2. 创建 HP 模型
model = TransformersModel(
    model_id='Qwen/Qwen3-4B',
    device_mesh=device_mesh,
    use_hyper_parallel=True,
)

# 3. kernelize（在 HP wrap 之后调用也可，因为 forward 替换对 DTensor 透明）
model = kernelize_model(model, mode='train', device='npu')
```

#### Function Kernel（monkey-patch 模块函数）

**结论：大多数场景兼容，需注意 HP 已注册的分布式算子。**

Function kernel 通过 `setattr(module, func_name, impl)` 替换模块级函数。HP 的 dispatch hook 通过 `__torch_function__` 协议注册在 DTensor 类型级别，不受 monkey-patch 影响：

```
场景 1: kernel 替换的是 HP 未注册的函数 (如 torch.nn.functional.silu)
  → monkey-patch 生效
  → 新函数接收 DTensor 输入
  → 如果新函数内部调用标准 torch ops → HP dispatch 自动处理
  → 兼容 ✓

场景 2: kernel 替换的是 HP 已注册的分布式算子 (如 torch.matmul)
  → monkey-patch 替换了 torch.matmul
  → HP 的 dispatch hook 仍通过 DTensor.__torch_function__ 触发
  → hook 内部调用 platform.get_op_name(op_call) 获取原始操作名
  → 如果 get_op_name 返回的是被替换后的函数名 → 可能不匹配 HP 注册表
  → 走 fallback 路径 → 功能正确但失去分片优化
  → 部分兼容 ⚠

场景 3: kernel 直接操作 tensor.data / tensor.storage()
  → DTensor 的 .data 是 local shard，非完整 tensor
  → 如果 kernel 假设是完整 tensor → 计算错误
  → 不兼容 ✗
```

#### 自定义 C++/Triton 算子（当前不存在，但为未来预留）

如果未来在 `csrc/` 下添加自定义算子（通过 `torch.library` 或 `torch.autograd.Function`）：

| 实现方式 | HP 兼容性 | 说明 |
|---------|----------|------|
| `torch.autograd.Function` | **兼容** | HP 保留自定义 autograd function，backward 自动传播 |
| `torch.library` / `TORCH_LIBRARY` | **需注册** | HP dispatch 未注册该 op → 走 fallback → 正确但无优化 |
| Triton kernel | **兼容** | Triton 在 tensor 级别操作，DTensor 自动转为 local tensor |

### 10.4 三种对接方式

#### 方式 1：直接使用（推荐，零改动）

Twinkle kernel 替换的 `forward()` 和函数都通过标准 torch ops 执行内部计算，HP 的 DTensor dispatch 自动处理分片。只需正常调用 `kernelize_model()` 即可：

```python
# 无需任何适配代码
model = kernelize_model(model, mode='train', device='npu')
# HP 的 fully_shard 分片 + Twinkle kernel 优化 → 两者透明共存
```

**前提条件**: kernel 内部必须通过标准 torch ops（`torch.matmul`, `F.linear` 等）执行计算，不直接操作 tensor 内部表示。

#### 方式 2：注册 HP 分布式算子（高级场景）

如果 kernel 使用了自定义 op 且需要分片感知（即该 op 的语义在不同 rank 上需要不同的分片策略）：

```python
from hyper_parallel.core.shard.ops.parallel_ops_register import register_distributed_op
from hyper_parallel.core.shard.ops.parallel_ops import DistributedOp

class MyKernelDistributedOp(DistributedOp):
    """将自定义 kernel 注册为 HP 分布式算子。"""
    def __init__(self):
        super().__init__("my_kernel_op_name")

    def infer_layout(self, layouts, extra_args=None):
        """定义输入→输出的分片布局推理规则。"""
        # 例如: 输入是 (Shard(0),) → 输出也是 (Shard(0),)
        ...

    def forward(self, local_args, local_kwargs, layouts):
        """在每个 rank 的 local shard 上执行 kernel。"""
        return my_kernel_forward(*local_args, **local_kwargs)

register_distributed_op("my_kernel_op_name", MyKernelDistributedOp())
```

**适用场景**: 自定义 attention kernel、自定义融合 matmul+activation 等。

#### 方式 3：SkipDTensorDispatch 绕过（调试/兼容场景）

如果 kernel 无法在 DTensor 环境下正确工作（如直接操作 tensor storage）：

```python
from hyper_parallel import SkipDTensorDispatch

# 方案 A: 完全绕过（kernel 在完整 tensor 上执行）
with SkipDTensorDispatch():
    output = custom_kernel(dtensor_input)  # DTensor 自动转为 local tensor

# 方案 B: 选择性绕过（指定某些 op 仍走 dispatch）
with SkipDTensorDispatch(no_skip={"torch.matmul"}):
    output = custom_kernel(input)  # 其他 op 走 dispatch, matmul 仍分片
```

**代价**: 绕过 dispatch 意味着在该上下文中失去分片优化，可能引入额外的 all-gather 通信。

### 10.5 冲突场景速查表

| 场景 | 是否冲突 | 原因 | 解决方案 |
|------|---------|------|---------|
| Kernel 调用标准 torch ops | **不冲突** | HP DTensor dispatch 自动拦截 | 直接使用 |
| Kernel 替换 Module.forward() | **不冲突** | HP 分片在参数级别，不修改 forward | 直接使用 |
| Kernel 使用 `torch.autograd.Function` | **不冲突** | HP 保留自定义 autograd function | 直接使用 |
| Kernel 替换 HP 已注册的分布式 op | **需适配** | monkey-patch 可能绕过 HP dispatch | 注册为 DistributedOp |
| Kernel 直接操作 `.data`/`.storage()` | **需适配** | DTensor 的 data 是 local shard | 使用 SkipDTensorDispatch |
| Kernel 使用 `torch.library` 自定义 op | **需适配** | HP 未注册 → 走 fallback（正确但无优化） | 注册为 DistributedOp |

---

## 11. HyperParallel × Twinkle 联合 RoadMap 与版本管理方案

### 11.1 版本号语义与发布节奏

#### 版本号约定

**联合发布规则**:

```
HP 0.x  + Twinkle 0.x  → 实验性集成, API 可变, 按需对齐
HP 1.0  + Twinkle 1.0  → 正式稳定 API, 严格向后兼容
```

**建议节奏**:

| 时间 | HP 版本 | Twinkle 版本 | 协调动作 |
|------|--------|-------------|---------|
| 2026 Q2 | 0.2.0 | 0.3.0 | 首个正式集成版本, 声明兼容范围 |
| 2026 Q3 | 0.3.0 | 0.4.0 | 多维混合并行能力扩展，自定义Kernel使能 |
| 2026 Q4 | 1.0.0-rc1 | 1.0.0-rc1 | MoE与多模态性能优化，API 冻结 |

#### 兼容性声明矩阵

在每个版本的 `setup.py` / `pyproject.toml` 中声明双向兼容范围：

```python
# HyperParallel setup.py (新增)
install_requires = [
    "numpy>=1.20.0",           # 放宽上限, 兼容 Twinkle 的 numpy>=2.0
    "torch>=2.6.0,<3.0.0",     # 显式声明
]

# Twinkle pyproject.toml (新增)
[project.optional-dependencies]
hyper_parallel = [
    "hyper_parallel>=0.2.0,<2.0.0",   # 声明最低 HP 版本
]
```

兼容性矩阵随每次发布更新，格式如下：

| HP \ Twinkle | 0.3.0 | 0.4.0 | 1.0.0 |
|-------------|-------|-------|-------|
| **0.1.0** | 实验性 (当前) | — | — |
| **0.2.0** | 兼容 | 兼容 | — |
| **0.3.0** | 兼容 | 兼容 | — |
| **1.0.0** | — | 兼容 | 兼容 |

### 11.2 多版本兼容性方案

#### 核心原则

1. **HP 永远不依赖 Twinkle 内部 API**: HP 仅通过 Twinkle 的公开策略接口 (`HyperParallelStrategy`, `DeviceMesh`) 交互，不直接引用 Twinkle 内部实现细节。
2. **Twinkle 通过 optional dependency 引入 HP**: `pip install twinkle-kit[hyper_parallel]` 是唯一安装路径，HP 不安装时 Twinkle 原生路径不受影响。
3. **版本范围声明优于固定版本**: 使用 `>=x.y,<z.0` 而非 `==x.y.z`，允许用户获取补丁更新。

#### 依赖版本协调策略

```
torch       :  HP >=2.6, Twinkle >=2.6  → 范围一致, 无冲突
numpy       :  HP >=1.20, Twinkle >=2.0 → Q2 统一为 >=1.20 (HP 去掉上限)
accelerate  :  HP 不声明, Twinkle any   → HP 不约束, 由 Twinkle 管理
transformers:  HP 不声明, Twinkle any   → HP 不约束, 由 Twinkle 管理
```

**版本冲突处理流程**:

```
用户报告依赖冲突
  │
  ├── pip check 输出
  │
  ├── 是否为 HP 声明约束导致?
  │   ├── YES → HP 侧修改 requirements.txt, 发版 patch release
  │   └── NO  → Twinkle 侧修改 pyproject.toml, 发版 patch release
  │
  └── 更新兼容性矩阵文档
```

#### API 变更管理

**不兼容变更清单** (需双方提前一个 minor 版本通知):

| 变更类型 | 通知方式 | 过渡期 |
|---------|---------|--------|
| 移除公开 API | DeprecationWarning + GitHub Issue | 至少 1 个 minor 版本 |
| 参数签名变更 | DeprecationWarning + 文档标注 | 至少 1 个 minor 版本 |
| 默认值变更 | Release Note 显式说明 | 至少 1 个 minor 版本 |
| 新增必需参数 | 先作为可选参数引入, 后续版本设为必需 | 至少 2 个 minor 版本 |

**对接接口稳定性承诺** (1.0 后):

```python
# 以下接口在 1.x 生命周期内保持不变:
twinkle.DeviceMesh                      # 构造参数不变
twinkle.model.transformers.TransformersModel(
    use_hyper_parallel=True,            # 开关参数不变
    hyper_parallel_config={...},         # 字典格式, 新增字段向后兼容
)
hyper_parallel.fully_shard(...)          # 核心分片 API 不变
hyper_parallel.SkipDTensorDispatch(...)  # 上下文管理器接口不变
```

### 11.3 联合门禁看护方案

```
L0 — Import 测试 (每次 PR)
  ├── import hyper_parallel + twinkle (无硬件依赖)
  ├── import torch_npu 条件处理
  └── 兼容性声明检查 (pip check)

L1 — 单元测试 (每次 PR)
  ├── HP fully_shard + Twinkle DeviceMesh 构造
  ├── HyperParallelStrategy.wrap_model
  └── SkipDTensorDispatch 上下文

L2 — 集成测试 (merge to main)
  ├── 整网级别验证
  └── Checkpoint 断点续训

```

#### 版本发布检查清单

每次 HP 或 Twinkle 发版前，必须通过以下检查：

- [ ] `pip check` 无依赖冲突
- [ ] L0 + L1 测试全部通过
- [ ] 兼容性矩阵更新
- [ ] CHANGELOG 包含影响对方的变更说明
- [ ] 跨版本升级测试: 前一版本 → 当前版本，无破坏性变更

### 11.5 沟通机制与社区 SIG 协作

#### 11.5.1 双团队直接沟通

| 渠道 | 频率 | 参与方 | 内容 |
|------|------|--------|------|
| GitHub Issue | 随时 | 双方社区 | Bug 报告, 功能请求 |
| 月度 Review | 每月 | 双方核心开发团队 | 进展与RoadMap 执行情况, 优先级调整 |
| 季度规划 | 每季度 | 双方团队 | 下一季度 RoadMap 制定 |
| 联合 Release | 按版本节奏 | 双方 Release Manager | 版本协调, 发布同步 |

#### 11.5.2 Parallel Training System SIG

HyperParallel 已纳入 **Parallel Training System SIG**（[SIG 主页](https://www.mindspore.cn/sig/Parallel%20Training%20System)），该 SIG 可以作为 HP 与 Twinkle 联动的关键社区纽带。

**SIG 定位**:

> 覆盖数据并行、半自动并行、自动并行、算子级并行、流水线并行、MoE 专家并行等分布式训练技术，支持 Ascend NPU / GPU / CPU 多硬件后端。

**SIG 在 HP × Twinkle 联动中的角色**:

```
┌──────────────────┐             ┌──────────────────┐
│ Parallel Training │◄── SIG ──►│ Twinkle Training  │
│ System SIG        │   联合    │ 社区              │
│                   │   议题    │                    │
│ · HyperParallel   │           │ · Twinkle-kit      │
│ · MindSpore 分布式│           │ · ModelScope       │
│ · CANN/HCCL       │           │ · transformers/peft│
└──────────────────┘             └──────────────────┘
        │                                │
        └──── 季度联合 SIG 会议 ─────────┘
              (双社区技术交流 + RoadMap 对齐)
```

**SIG 可提供的协作资源**:

| 资源 | 说明 | 对 HP × Twinkle 的价值 |
|------|------|----------------------|
| 双周 SIG 例会 | SIG 成员定期技术分享和议题讨论 | HP 团队可直接在 SIG 会议上汇报 Twinkle 集成进展，获取 MindSpore 社区反馈 |
| SIG 邮件列表 / 微信群 | SIG 成员异步沟通渠道 | 跨社区问题快速响应，不依赖双方私下沟通 |
| SIG 代码仓库 | `gitcode.com/mindspore/hyper-parallel` | HP 的 PR review 和 release 流程已与 SIG 治理绑定 |

**建议的 SIG 联动节奏**:

| 时间 | 动作 | 目标 |
|------|------|------|
| 2026 Q2 | HP 团队在 SIG 例会上做 "HyperParallel × Twinkle 集成方案" 专题报告 | 向 MindSpore 社区同步跨生态合作方向，收集分布式训练 SIG 成员的意见 |
| 2026 Q2 | 邀请 Twinkle 核心开发者作为 SIG Guest 参加一次例会 | 建立双方社区的正式联系 |
| 2026 Q3 | SIG 例会增设 "跨生态集成" 常设议题 (每季度一次) | 持续跟踪 HP × Twinkle 的 GPU 适配、Kernel 协同等进展 |
| 2026 Q3 | 在 SIG 内发起 "GPU 后端验证" 任务组 | 动员 SIG 社区中有 GPU 资源的成员参与验证 |、

#### 13.5.3 跨社区 Issue 协作规范

为避免两个社区间 Issue 碎片化，建议采用以下规范：

| Issue 类型 | 提报位置 | 标签 | 处理方 |
|-----------|---------|------|--------|
| HP 内部 Bug | `gitcode.com/mindspore/hyper-parallel/issues` | `bug` | HP 团队 |
| Twinkle 内部 Bug | `github.com/modelscope/twinkle/issues` | `bug` | Twinkle 团队 |
| 集成问题 (HP+Twinkle) | 双方均可, 交叉引用 | `area:hyper-parallel` (Twinkle) / `area:twinkle` (HP) | 双方协同 |
| 依赖冲突 | 优先提报到 HP 侧 | `type:dependency` | HP 团队 (通常需 HP 放宽约束) |
| 功能请求 (跨生态) | SIG 邮件列表 / 双周例会 | — | SIG 讨论 → 分配到具体仓库 |