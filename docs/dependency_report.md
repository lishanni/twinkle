# Twinkle + HyperParallel Python 三方组件依赖分析报告

> **环境**: conda env `hyper_twinkle` on openEuler 22.03 LTS-SP1 (aarch64)
> **Python**: 3.11.15 | **硬件**: Ascend 910B3 × 8
> **验证时间**: 2026-05-06 | **基于 commit**: `d001934`

---

## 1. 执行摘要

当前 `hyper_twinkle` conda 环境共安装 **169 个包**。核心依赖链为：

```
twinkle-kit 0.3.0.dev0
├── numpy >=2.0.0,<2.3.0              ← 声明
├── datasets, omegaconf, fastapi
├── modelscope[framework] >=1.34.0
├── safetensors, peft >=0.11.0, transformers
├── [transformers extra] accelerate, torch >=2.6.0, torchvision
└── [hyper_parallel extra] hyper_parallel

hyper_parallel 0.1.0
├── numpy >=1.20.0,<2.0.0             ← 声明 (与 Twinkle 冲突!)
└── [隐式] torch, torch_npu, CANN hccl
```

**关键问题**:

| # | 问题 | 严重程度 | 状态 |
|---|------|----------|------|
| 1 | numpy 版本冲突 (Twinkle ≥2.0 vs HP <2.0) | **Critical** | 当前 numpy=2.2.6 违反 HP 声明 |
| 2 | torch_npu 双重注册导致 import 崩溃 | **Critical** | 需 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |
| 3 | NVIDIA CUDA 包在 aarch64 NPU 机器上无效 | **Low** | 15 个 nvidia-* 包为死重 |
| 4 | HyperParallel 声明依赖不完整 | **Medium** | 仅声明 numpy，隐式依赖 torch/torch_npu |
| 5 | CANN 系统包不在 pip 管理范围内 | **Info** | hccl/te 等 19 个包来自 CANN 8.5.0 |

---

## 2. 核心依赖版本矩阵

### 2.1 直接依赖（声明 + 实际安装）

| 包名 | Twinkle 声明范围 | HP 声明范围 | 实际版本 | 满足 Twinkle | 满足 HP |
|------|-----------------|------------|---------|-------------|---------|
| numpy | ≥2.0.0,<2.3.0 | ≥1.20.0,<2.0.0 | **2.2.6** | YES | **NO** |
| torch | ≥2.6.0,<3.0.0 | (隐式) | **2.7.1+cpu** | YES | N/A |
| transformers | any | (隐式) | **5.6.2** | YES | N/A |
| accelerate | any | (隐式) | **1.11.0** | YES | N/A |
| peft | ≥0.11.0,≤0.19.0 | (隐式) | **0.18.1** | YES | N/A |
| datasets | any | (无) | **4.8.4** | YES | N/A |
| modelscope | [framework]≥1.34.0 | (无) | **1.35.3** | YES | N/A |
| omegaconf | ≥2.3.0,<3.0.0 | (无) | **2.3.0** | YES | N/A |
| safetensors | any | (隐式) | **0.7.0** | YES | N/A |
| fastapi | any | (无) | **0.135.3** | YES | N/A |

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
| `modelscope` | ERR | 1.35.3 | RuntimeError: torch_npu extension |
| `omegaconf` | OK | 2.3.0 | |
| `fastapi` | OK | 0.135.3 | |
| `hccl` | OK | 0.1.0 | 来自 CANN 8.5.0 |
| `megatron` | FAIL | — | 未安装 |
| `transformer_engine` | FAIL | — | 未安装 |
| `vllm` | FAIL | — | 未安装 |
| `ray` | FAIL | — | 未安装 |
| `tinker` | FAIL | — | 未安装 |

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

### 3.3 pip check 发现的问题

```
hyper-parallel 0.1.0 has requirement numpy<2.0.0,>=1.20.0, but you have numpy 2.2.6
nvidia-cusparselt-cu13 0.8.0 is not supported on this platform
op-compile-tool 0.1.0 requires getopt, which is not installed  ← 误报 (stdlib)
op-compile-tool 0.1.0 requires inspect, which is not installed  ← 误报 (stdlib)
op-compile-tool 0.1.0 requires multiprocessing, which is not installed  ← 误报 (stdlib)
```

---

## 4. numpy 版本冲突详细分析

这是当前环境最严重的依赖冲突：

| 项目 | 声明 | 原因 |
|------|------|------|
| Twinkle (`pyproject.toml`) | `numpy>=2.0.0,<2.3.0` | Twinkle 代码使用了 numpy 2.x API |
| HyperParallel (`requirements.txt`) | `numpy>=1.20.0,<2.0.0` | HP 库可能依赖 numpy 1.x 的某些行为 |

**冲突范围**: numpy 2.0.0 是分歧点 — Twinkle 要求 ≥2.0.0，HP 要求 <2.0.0，二者无交集。

**当前状态**: numpy 2.2.6 满足 Twinkle 但违反 HP。实际运行中 HP 似乎兼容 numpy 2.x（未观察到运行时错误），但这是一个潜在的风险点。

**解决选项**:

| 方案 | 可行性 | 风险 |
|------|--------|------|
| A. 放宽 HP 的 numpy 上限为 `<2.3.0` | 需 HP 团队配合 | 需验证 HP 所有功能 |
| B. 降低 Twinkle 的 numpy 下限为 `>=1.20.0` | 需修改 Twinkle | Twinkle 可能依赖 2.x API |
| C. 保持现状 (numpy=2.2.6) | 零改动 | HP 未声明兼容性，有潜在风险 |

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

## 6. NVIDIA CUDA 包（当前机器无效）

当前 aarch64 NPU 环境中安装了 **15 个 nvidia-* 包**，这些包来自 PyTorch 的 CUDA 依赖但在此平台上**完全无用**：

| 包名 | 版本 | 说明 |
|------|------|------|
| nvidia-cublas | 13.1.0.3 | CUDA 基础线性代数 |
| nvidia-cuda-cupti | 13.0.85 | CUDA profiling |
| nvidia-cuda-nvrtc | 13.0.88 | CUDA runtime compiler |
| nvidia-cuda-runtime | 13.0.96 | CUDA runtime |
| nvidia-cudnn-cu13 | 9.19.0.56 | cuDNN |
| nvidia-cufft | 12.0.0.61 | FFT |
| nvidia-cufile | 1.15.1.6 | cuFile |
| nvidia-curand | 10.4.0.35 | 随机数 |
| nvidia-cusolver | 12.0.4.66 | 线性代数求解 |
| nvidia-cusparse | 12.6.3.3 | 稀疏矩阵 |
| nvidia-cusparselt-cu13 | 0.8.0 | 稀疏矩阵 (不支持此平台) |
| nvidia-nccl-cu13 | 2.28.9 | NCCL 通信库 |
| nvidia-nvjitlink | 13.0.88 | JIT linker |
| nvidia-nvshmem-cu13 | 3.4.5 | NVSHMEM |
| nvidia-nvtx | 13.0.85 | NVTX profiling |

**额外包**: `cuda-bindings 13.2.0`, `cuda-pathfinder 1.5.2`, `cuda-toolkit 13.0.2`, `triton 3.6.0`

**影响**: 占用约 2-3GB 磁盘空间，对运行时无影响（torch 是 CPU 构建，不加载 CUDA）。

**建议**: 在 NPU-only 环境中可安全移除，但需注意 pip 依赖约束可能阻止卸载。

---

## 7. CANN 系统级 Python 包

CANN 8.5.0 安装在 `/home/lsn/Ascend/cann-8.5.0/`，提供以下 Python 包（不在 pip 管理范围）：

| 包名 | 版本 | 功能 |
|------|------|------|
| hccl | 0.1.0 | 华为集合通信库 (对应 NCCL) |
| te | 0.4.0 | TurboEngine 算子加速 |
| acl | — | Ascend Computing Language |
| asc_op_compile_base | 0.1.0 | 算子编译基础库 |
| asc_opc_tool | 0.1.0 | 算子编译工具 |
| auto_tune | 0.1.0 | 自动调优 |
| dataflow | 0.0.1 | 数据流引擎 |
| es_math | 1.0.0 | 数学库 |
| ge_py | 0.0.1 | Graph Engine |
| llm_datadist | 0.1.0 | LLM 数据分发 |
| llm_datadist_v1 | 0.0.1 | LLM 数据分发 v1 |
| mspti | 0.0.1 | Profiling 工具 |
| msobjdump | 0.1.0 | 对象转储 |
| op_compile_tool | 0.1.0 | 算子编译工具 |
| opc_tool | 0.1.0 | OPC 工具 |
| op_gen | 0.1 | 算子生成 |
| op_test_frame | 0.1 | 算子测试框架 |
| schedule_search | 0.0.1 | 调度搜索 |
| superkernel | 0.1.0 | 超级内核 |
| show_kernel_debug_data | 0.1.0 | 内核调试数据 |

**特点**:
- 安装路径: `/home/lsn/Ascend/cann-8.5.0/python/site-packages/`
- 不由 pip 管理，通过 CANN 安装器部署
- GPU 环境下完全不需要这些包

---

## 8. Twinkle 可选依赖安装状态

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

## 9. 完整依赖关系图

### 9.1 Twinkle 直接 + 传递依赖 (已安装)

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
    ├── numpy (声明: >=1.20.0,<2.0.0)  ← 冲突
    └── [隐式] torch, torch_npu, transformers
```

### 9.2 平台/硬件层

```
硬件: Ascend 910B3 × 8 (aarch64)
├── CANN 8.5.0 (/home/lsn/Ascend/)
│   ├── hccl 0.1.0 (通信库)
│   ├── te 0.4.0 (算子加速)
│   ├── acl (计算语言)
│   └── 19 个其他 Python 包
├── torch_npu 2.7.1 (NPU bridge)
│   └── torch 2.7.1+cpu
└── [未使用] nvidia-cublas/nccl/cudnn/... (CUDA, aarch64 不可用)
```

---

## 10. GPU 环境迁移依赖差异

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
| numpy 冲突 | 存在 | 相同 |

---

## 11. 总结与建议

### 必须解决

1. **numpy 版本冲突**: Twinkle (≥2.0) 与 HyperParallel (<2.0) 无交集。建议 HP 团队更新 `requirements.txt` 放宽上限，或在 Twinkle 端降级。
2. **torch_npu 双重注册**: 必须在所有脚本/服务入口设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。

### 建议改进

3. **HyperParallel 声明依赖不完整**: `requirements.txt` 仅声明 numpy，应添加 `torch` 和 `torch_npu`（条件）。
4. **清理无用 NVIDIA 包**: 当前 aarch64 NPU 环境中的 15 个 nvidia-* 包和 cuda-* 包为死重。
5. **Import 顺序文档化**: 在 Twinkle 文档中说明 `import torch; import torch_npu` 必须在 `from peft import ...` 和 `from transformers import ...` 之前。

### 可选优化

6. 将 CANN 包纳入 conda 环境管理（目前依赖系统路径）。
7. 考虑将 `torch` 固定为 CPU 构建以减少 NPU 环境中的依赖体积。

---

## 12. Twinkle Kernel 优化算子与 HyperParallel 兼容性分析

### 12.1 Twinkle Kernel 模块架构

`src/twinkle/kernel/` **不是自定义 C++/CUDA 算子实现**，而是一个围绕 HuggingFace `kernels` 包的**编排调度层**。模块结构：

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

### 12.2 HyperParallel 的算子调度机制

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

HP 内置 **40+ 分布式算子**（matmul, embedding, attention 等），通过 `parallel_ops.py` 中的 `DistributedOp` 基类自动注册。

### 12.3 兼容性矩阵：Twinkle Kernel × HyperParallel

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
| Custom 通信 (all_reduce 等) | **冲突** | HP 的 comm_fusion 可能覆盖自定义通信模式 |

### 12.4 三种对接方式

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

### 12.5 冲突场景速查表

| 场景 | 是否冲突 | 原因 | 解决方案 |
|------|---------|------|---------|
| Kernel 调用标准 torch ops | **不冲突** | HP DTensor dispatch 自动拦截 | 直接使用 |
| Kernel 替换 Module.forward() | **不冲突** | HP 分片在参数级别，不修改 forward | 直接使用 |
| Kernel 使用 `torch.autograd.Function` | **不冲突** | HP 保留自定义 autograd function | 直接使用 |
| Kernel 替换 HP 已注册的分布式 op | **部分冲突** | monkey-patch 可能绕过 HP dispatch | 注册为 DistributedOp |
| Kernel 直接操作 `.data`/`.storage()` | **冲突** | DTensor 的 data 是 local shard | 使用 SkipDTensorDispatch |
| Kernel 使用自定义通信 (all_reduce 等) | **冲突** | HP 的 comm_fusion 会覆盖 | 关闭 comm_fusion |
| Kernel 使用 `torch.library` 自定义 op | **需适配** | HP 未注册 → 走 fallback（正确但无优化） | 注册为 DistributedOp |

### 12.6 当前环境限制

1. **`kernels` 包未安装**: 当前 conda 环境中没有安装 HuggingFace `kernels` 包，Twinkle kernel 模块的所有注册和应用操作均为空操作。需 `pip install kernels` 才能启用。
2. **无实际 kernel 实现可验证**: 由于 `kernels` 包未安装，无法在当前环境验证 kernel 与 HP 的实际交互行为。以上分析基于代码架构推导。
3. **建议验证步骤**:
   - 安装 `kernels` 包: `pip install kernels`
   - 注册一个 layer kernel 并 kernelize 模型
   - 在 HP FSDP2 路径下运行训练，对比 loss 数值与无 kernel 基线
   - 检查是否有 DTensor 相关的 warning/error

---

## 13. HyperParallel × Twinkle 联合 RoadMap 与版本管理方案

> **视角**: HyperParallel 开发团队，面向 Twinkle 社区协作
> **规划周期**: 2026 Q2–Q4（季度级别）
> **当前版本**: HP 0.1.0 | Twinkle 0.3.0.dev0

### 13.1 版本号语义与发布节奏

#### 版本号约定

两个项目均采用 **SemVer** (Semantic Versioning)，但含义有侧重：

| 项目 | 格式 | 含义 | 示例 |
|------|------|------|------|
| HyperParallel | `MAJOR.MINOR.PATCH` | MAJOR=不兼容 API 变更, MINOR=向后兼容新功能 | `0.2.0` |
| Twinkle | `MAJOR.MINOR.PATCH[.devN]` | 同上, dev 表示开发版 | `0.3.0.dev0` |

**联合发布规则**:

```
HP 0.x  + Twinkle 0.x  → 实验性集成, API 可变, 按需对齐
HP 1.0  + Twinkle 1.0  → 正式稳定 API, 严格向后兼容
```

**建议节奏**:

| 时间 | HP 版本 | Twinkle 版本 | 协调动作 |
|------|--------|-------------|---------|
| 2026 Q2 | 0.2.0 | 0.3.0 | 首个正式集成版本, 声明兼容范围 |
| 2026 Q3 | 0.3.0 | 0.4.0 | GPU 支持, Kernel 协同 |
| 2026 Q4 | 1.0.0-rc1 | 1.0.0-rc1 | API 冻结, 联合稳定性验证 |

#### 兼容性声明矩阵

在每个版本的 `setup.py` / `pyproject.toml` 中声明双向兼容范围：

```python
# HyperParallel setup.py (建议新增)
install_requires = [
    "numpy>=1.20.0",           # 放宽上限, 兼容 Twinkle 的 numpy>=2.0
    "torch>=2.6.0,<3.0.0",     # 显式声明
]

# Twinkle pyproject.toml (建议新增)
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

### 13.2 季度 RoadMap

---

#### 2026 Q2: 基础设施对齐（"能跑起来"）

**目标**: 解决当前已知阻塞性问题，建立协作基础设施。

**HP 侧 (v0.2.0)**:

| 优先级 | 工作项 | 交付物 | 验收标准 |
|--------|--------|--------|---------|
| P0 | numpy 版本上限放宽 | `requirements.txt`: `numpy>=1.20.0` (去掉 `<2.0.0`) | `pip check` 无冲突 |
| P0 | 显式声明 torch 依赖 | `install_requires` 添加 `torch>=2.6.0,<3.0.0` | 新环境 `pip install hyper_parallel` 可正确拉取 torch |
| P1 | 设备后端抽象层 | `get_device_handle()` 默认值从 `"npu"` 改为动态检测 | GPU 环境下不报错 |
| P1 | symmetric_memory Stream 抽象 | `torch.npu.Stream()` → `torch.cuda.Stream()` 或动态选择 | GPU 上不崩溃 |
| P2 | CI 集成测试 | 新增 `tests/torch/test_twinkle_integration.py` | Twinkle HP 路径基础 forward/backward 通过 |

**Twinkle 侧 (v0.3.0 稳定版)**:

| 优先级 | 工作项 | 交付物 | 验收标准 |
|--------|--------|--------|---------|
| P0 | `hyper_parallel` extra 版本声明 | `pyproject.toml` 添加版本范围约束 | `pip install twinkle-kit[hyper_parallel]` 正确解析 |
| P1 | torch_npu import 顺序文档 | README/CONTRIBUTING 说明 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` | 新用户不再踩坑 |
| P1 | `log_npu_memory` 重构 | 重命名为 `log_device_memory()`, 支持 GPU/NPU | 两个平台均输出内存信息 |

**联合工作**:

| 工作项 | 负责方 | 交付物 |
|--------|--------|--------|
| 兼容性矩阵文档 | HP + Twinkle | `COMPATIBILITY.md` — 版本对应关系 + 已知限制 |
| 联合 Issue 模板 | Twinkle | GitHub issue template 含 "HP 相关" 标签 |
| 周同步会议 | 双方 | 每周 30 min, 跟踪阻塞项 |

---

#### 2026 Q3: 跨平台扩展（"GPU 也能跑"）

**目标**: HyperParallel 在 NVIDIA GPU 上可用，Twinkle Kernel 与 HP 协同工作。

**HP 侧 (v0.3.0)**:

| 优先级 | 工作项 | 交付物 | 验收标准 |
|--------|--------|--------|---------|
| P0 | GPU FSDP2 验证 | 所有 cookbook 脚本提供 `*_gpu.py` 版本 | 4×A100 上 forward/backward/step 通过 |
| P0 | Cookbook 设备动态化 | `torch.npu.device_count()` → 动态检测 | GPU/NPU 自动适配 |
| P1 | Profiler 双平台 | 条件导入 `torch_npu.profiler` / `torch.profiler` | GPU profiling trace 可用 |
| P1 | NCCL 通信路径验证 | `comm_fusion` + `enable_prefetch` 在 NCCL 上工作 | GPU 8卡训练无 hang |
| P2 | symmetric_memory CUDA 版本 | `build_symmetric_memory.sh` 支持 CUDA 编译 | GPU 上可启用 comm_fusion |
| P2 | Expert Parallel GPU 路径 | `npu_grouped_matmul` → `torch.bmm` 或 `grouped_gemm` fallback | MoE 模型 GPU 可训练 |

**Twinkle 侧 (v0.4.0)**:

| 优先级 | 工作项 | 交付物 | 验收标准 |
|--------|--------|--------|---------|
| P0 | HP 策略 GPU 测试 | CI 新增 GPU + HP FSDP2 测试用例 | GitHub CI GPU runner 通过 |
| P1 | Kernel × HP 集成测试 | `tests/test_kernel_hp_integration.py` | kernelize + fully_shard 联合通过 |
| P2 | Performance benchmark 框架 | 统一的 NPU/GPU 对比 benchmark 脚本 | 吞吐量 + 数值一致性报告 |

**联合工作**:

| 工作项 | 负责方 | 交付物 |
|--------|--------|--------|
| GPU 兼容性联合报告 | HP + Twinkle | 基于实际 GPU 集群验证的完整兼容性报告 |
| 性能基线 (Baseline) | HP | NPU vs GPU 上 HP FSDP2 吞吐量对比 (4B/32B 模型) |
| 联合 Blog / Tutorial | 双方 | "Twinkle + HyperParallel 分布式训练指南" |

---

#### 2026 Q4: 稳定化与 1.0（"生产可用"）

**目标**: API 冻结，全面稳定性验证，发布 1.0 候选版本。

**HP 侧 (v1.0.0-rc1)**:

| 优先级 | 工作项 | 交付物 | 验收标准 |
|--------|--------|--------|---------|
| P0 | API 冻结 | 所有公开 API 标记为 stable, 移除实验性标记 | API 不再变更 |
| P0 | 多版本兼容性测试矩阵 | torch 2.6/2.7 × numpy 1.x/2.x × NPU/GPU | 全部组合通过 CI |
| P1 | 文档完善 | API reference, migration guide, FAQ | 覆盖所有公开接口 |
| P1 | 性能回归测试 | CI 集成性能 benchmark, 自动对比基线 | 性能无退化 |
| P2 | MindSpore 后端对齐 | MindSpore 平台与 Twinkle 对接探索 | 概念验证 |

**Twinkle 侧 (v1.0.0-rc1)**:

| 优先级 | 工作项 | 交付物 | 验收标准 |
|--------|--------|--------|---------|
| P0 | HP 集成 API 稳定化 | `HyperParallelStrategy` 公开 API 冻结 | 1.0 内不变更 |
| P0 | 多后端 CI | GitHub CI: NPU (self-hosted) + GPU (cloud runner) | 每次提交自动验证 |
| P1 | 端到端测试覆盖 | Qwen3-4B/32B LoRA 全流程: 加载→训练→保存→恢复 | 全流程无手动干预 |
| P2 | Kernel 正式集成 | kernel + HP 组合成为推荐配置 | 文档+示例齐全 |

**联合工作**:

| 工作项 | 负责方 | 交付物 |
|--------|--------|--------|
| 1.0.0 联合 Release Note | 双方 | 里程碑文档, 包含所有已解决问题清单 |
| 生产环境部署指南 | HP | NPU 集群 + GPU 集群最佳实践 |
| 社区反馈收集 | Twinkle | GitHub Discussion 线程, 收集早期用户问题 |

### 13.3 多版本兼容性方案

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

### 13.4 CI/CD 联合方案

#### CI 矩阵

| 维度 | 组合 | 触发条件 |
|------|------|---------|
| torch | 2.6, 2.7 | 每次 PR |
| numpy | 1.26, 2.2 | 每次 PR |
| 设备 | NPU (self-hosted), GPU (cloud) | merge to main |
| HP 版本 | latest release, main branch | 每日 cron |
| Twinkle 版本 | latest release, main branch | 每日 cron |

#### 联合测试用例分层

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
  ├── Qwen3-0.6B LoRA + HP FSDP2, 2 卡, 10 步
  ├── 数值一致性: HP vs Native FSDP2 (avg |diff| < 0.01)
  └── Checkpoint 保存/加载往返

L3 — 端到端测试 (每日 cron)
  ├── Qwen3-4B LoRA 完整训练, 4 卡
  ├── 性能回归: 吞吐量不低于基线 95%
  └── Kernel + HP 联合验证 (如果 kernels 包可用)
```

#### 版本发布检查清单

每次 HP 或 Twinkle 发版前，必须通过以下检查：

- [ ] `pip check` 无依赖冲突
- [ ] L0 + L1 测试全部通过
- [ ] 兼容性矩阵更新
- [ ] CHANGELOG 包含影响对方的变更说明
- [ ] 跨版本升级测试: 前一版本 → 当前版本，无破坏性变更

### 13.5 沟通机制与社区 SIG 协作

#### 13.5.1 双团队直接沟通

| 渠道 | 频率 | 参与方 | 内容 |
|------|------|--------|------|
| GitHub Issue | 随时 | 双方社区 | Bug 报告, 功能请求 |
| 周同步会 | 每周 | HP + Twinkle 核心 2-3 人 | 阻塞项, 进度同步 |
| 月度 Review | 每月 | 双方团队 | RoadMap 执行情况, 优先级调整 |
| 季度规划 | 每季度 | 双方团队 | 下一季度 RoadMap 制定 |
| 联合 Release | 按版本节奏 | 双方 Release Manager | 版本协调, 发布同步 |

#### 13.5.2 MindSpore Parallel Training System SIG

HyperParallel 是 MindSpore 社区孵化项目，已纳入 **Parallel Training System SIG**（[SIG 主页](https://www.mindspore.cn/sig/Parallel%20Training%20System)）。该 SIG 是 HP 与 Twinkle 联动的关键社区纽带。

**SIG 定位**:

> 覆盖数据并行、半自动并行、自动并行、算子级并行、流水线并行、MoE 专家并行等分布式训练技术，支持 Ascend NPU / GPU / CPU 多硬件后端。

**SIG 在 HP × Twinkle 联动中的角色**:

```
MindSpore 社区                    ModelScope 社区
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
| SIG Chair 协调 | SIG Chair 负责跨社区协调 | 当 HP × Twinkle 出现架构分歧时，SIG Chair 可提供中立调解 |
| 昇腾硬件资源 | SIG 成员可申请 NPU 测试资源 | 确保 HP × Twinkle 在多代昇腾硬件上的持续验证 |

**建议的 SIG 联动节奏**:

| 时间 | 动作 | 目标 |
|------|------|------|
| 2026 Q2 | HP 团队在 SIG 例会上做 "HyperParallel × Twinkle 集成方案" 专题报告 | 向 MindSpore 社区同步跨生态合作方向，收集分布式训练 SIG 成员的意见 |
| 2026 Q2 | 邀请 Twinkle 核心开发者作为 SIG Guest 参加一次例会 | 建立双方社区的正式联系 |
| 2026 Q3 | SIG 例会增设 "跨生态集成" 常设议题 (每季度一次) | 持续跟踪 HP × Twinkle 的 GPU 适配、Kernel 协同等进展 |
| 2026 Q3 | 在 SIG 内发起 "GPU 后端验证" 任务组 | 动员 SIG 社区中有 GPU 资源的成员参与验证 |
| 2026 Q4 | SIG 例会进行 HP × Twinkle 1.0 联合发布预演 | 收集 MindSpore 社区对 API 稳定性的最终反馈 |

#### 13.5.3 跨社区 Issue 协作规范

为避免两个社区间 Issue 碎片化，建议采用以下规范：

| Issue 类型 | 提报位置 | 标签 | 处理方 |
|-----------|---------|------|--------|
| HP 内部 Bug | `gitcode.com/mindspore/hyper-parallel/issues` | `bug` | HP 团队 |
| Twinkle 内部 Bug | `github.com/modelscope/twinkle/issues` | `bug` | Twinkle 团队 |
| 集成问题 (HP+Twinkle) | 双方均可, 交叉引用 | `area:hyper-parallel` (Twinkle) / `area:twinkle` (HP) | 双方协同 |
| 依赖冲突 | 优先提报到 HP 侧 | `type:dependency` | HP 团队 (通常需 HP 放宽约束) |
| 功能请求 (跨生态) | SIG 邮件列表 / 双周例会 | — | SIG 讨论 → 分配到具体仓库 |

### 13.6 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| HP numpy 放宽上限后出现兼容性问题 | 中 | 训练结果错误 | Q2 建立 numpy 1.x/2.x 双版本 CI |
| torch 版本升级 (2.8+) 破坏 HP 内部 API | 高 | 集成不可用 | HP 维护 torch 版本兼容层, 提前在 nightly 测试 |
| 双方 API 设计理念冲突 | 中 | 集成方案反复修改 | Q2 冻结对齐接口, 后续只扩展不修改 |
| GPU 验证资源不足 | 中 | GPU 路径质量不确定 | Q3 申请云 GPU 资源, 或寻求社区 GPU 贡献者 |
| 双方发版节奏不同步 | 低 | 用户拿到不兼容组合 | 兼容性矩阵 + `pip install twinkle-kit[hyper_parallel]` 约束 |
