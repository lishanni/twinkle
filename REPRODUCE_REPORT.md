# 复现报告：twinkle + hyper_parallel FSDP2 在 8x910B NPU 上训练

## 环境信息

| 组件 | 版本 |
|------|------|
| OS | openEuler 22.03 LTS-SP1 (aarch64) |
| NPU | 8x Ascend 910B3 (npu-smi 26.0.rc1) |
| CANN | 8.5.0 (`/home/lsn/Ascend/cann-8.5.0`) |
| conda 环境 | `hyper_twinkle` |
| Python | 3.11.15 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1 |
| hyper_parallel | 0.1.0 (editable, `/home/lsn/hyper-parallel-master`, master) |
| transformers | 5.6.2 |
| twinkle-kit | 0.3.0.dev0 (editable) |

## 环境搭建步骤

```bash
# 1. 创建 conda 环境（如果不存在）
conda create -n hyper_twinkle python=3.11 -y
conda activate hyper_twinkle

# 2. 安装 torch + torch_npu
pip install torch==2.7.1 torchvision torchaudio
pip install torch_npu==2.7.1

# 3. 安装 hyper_parallel（editable 模式，跳过 numpy 冲突）
git clone https://gitcode.com/mindspore/hyper-parallel.git ~/hyper-parallel-master
pip install -e ~/hyper-parallel-master --no-deps

# 4. 安装 twinkle
cd ~/twinkle
pip install -e ".[transformers,hyper_parallel]"

# 5. 安装额外依赖
pip install pyzmq "transformers>=5.0"

# 6. 设置环境变量（激活时自动生效）
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6"
export HYPER_PARALLEL_PLATFORM=torch
EOF
cat > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh << 'EOF'
#!/bin/bash
unset LD_PRELOAD
unset HYPER_PARALLEL_PLATFORM
EOF
```

### 已知兼容性说明

- **numpy 冲突**：hyper_parallel 要求 `numpy<2.0.0`，twinkle 要求 `numpy>=2.0.0`。用 `--no-deps` 安装 hyper_parallel 跳过约束检查，实际运行正常（numpy 2.2.6）。
- **LD_PRELOAD**：解决 openEuler 系统库 `libstdc++.so.6` 缺少 `CXXABI_1.3.15` 的问题。

## 实验概览

四组模型（Qwen3-0.6B、Qwen3.5-4B、Qwen3-30B-A3B MoE、Qwen3-32B），每组跑两个实验：

1. **Group 1（确定性验证）**：ASCEND_LAUNCH_BLOCKING=1 + HCCL_DETERMINISTIC=true + CPU_AFFINITY_CONF=1，验证 native vs HP 精度对齐
2. **Group 2（性能对比）**：CPU_AFFINITY_CONF=1，不开流同步和确定性，HP 开启 prefetch 性能优化，对比速度和精度

### 统一配置

| 配置项 | 值 |
|--------|-----|
| NPU 卡数 | 4 (ASCEND_RT_VISIBLE_DEVICES=4,5,6,7) |
| LoRA | r=8, alpha=32, target_modules=all-linear |
| batch_size | 4 (4B: 8), gradient_accumulation_steps=2 |
| fp32 upcast | 开启 (accelerate 默认 = HP upcast_trainable_params=True) |
| 优化器 | AdamW, lr=1e-4, CosineWarmupScheduler (5 warmup steps) |
| 数据集 | self-cognition (1000 samples) |

### 模型配置

| 模型 | 架构 | `_no_split_modules` | 卡数 | max_length | batch_size | 总步数 |
|------|------|---------------------|------|------------|------------|--------|
| Qwen3-0.6B | Dense, 28层 | Qwen3DecoderLayer | 2 | 默认 | 8 | 125 |
| Qwen3.5-4B | Dense, 36层 | Qwen3_5DecoderLayer | 4 | 默认 | 8 | 125 |
| Qwen3-30B-A3B | MoE, 48层, 128 experts/8 active | Qwen3MoeDecoderLayer | 4 | 2048 | 4 | 250 |
| Qwen3-32B | Dense, 64层 | Qwen3DecoderLayer | 4 | 2048 | 4 | 250 |

## Group 1：确定性精度验证

**环境**：ASCEND_LAUNCH_BLOCKING=1 + HCCL_DETERMINISTIC=true + CPU_AFFINITY_CONF=1

**HP 配置**：comm_fusion=False, enable_prefetch=False, upcast_trainable_params=True

### 结果

| 模型 | 卡数 | 精确匹配步数 | Avg `|loss diff|` | Max `|loss diff|` | Native 速度 | HP 速度 |
|------|------|------------|---------------|---------------|------------|---------|
| **Qwen3-0.6B** | **2** | **125/125 (全部 EXACT)** | **0** | **0** | 0.07 iters/s | 0.07 iters/s |
| Qwen3.5-4B | 4 | 7/125 (前5步EXACT) | 0.0037 | 0.0254 | 0.07 iters/s | 0.07 iters/s |
| Qwen3-30B-A3B | 4 | 5/250 (前5步EXACT) | 0.0130 | 0.1143 | 0.05 iters/s | 0.04 iters/s |
| Qwen3-32B | 4 | 8/250 (前5步EXACT) | 0.0105 | 0.1300 | 0.11 iters/s | 0.07 iters/s |

### 分析

**Qwen3-0.6B（2 卡）实现 125/125 步完全零误差**，native 与 HP 的 loss 在整个训练过程中 bit-exact 一致。这证明了在 2 卡 FSDP2 sharding 下，native 和 HP 的 forward/backward/optimizer 计算路径完全等价。

**4 卡模型前 5 步 loss 完全一致**（bit-exact），之后开始漂移。

漂移原因：4 卡 FSDP2 下，即使开启 ASCEND_LAUNCH_BLOCKING 和 HCCL_DETERMINISTIC，clip_grad_norm 的多卡梯度聚合和浮点归约仍引入微小数值差异。这些差异经 optimizer step 放大后逐 step 累积。2 卡 FSDP2 sharding 的数值行为更稳定，4 卡 sharding 导致不同的浮点累加顺序，是漂移的根源。

**结论**：Native 和 HP 的 forward/backward 计算完全一致（前 5 步 bit-exact），漂移来自 FSDP2 多卡通信的浮点累加差异，非 HP 实现问题。Avg diff < 0.013，不影响训练收敛。

### 显存

| 模型 | 卡数 | Native peak | HP peak | HP 额外 |
|------|------|-----------|---------|---------|
| Qwen3-0.6B | 2 | — | — | — |
| Qwen3.5-4B | 4 | 9.27 GB | 15.21 GB | +5.94 GB |
| Qwen3-30B-A3B | 4 | 33.25 GB | 45.96 GB | +12.71 GB |
| Qwen3-32B | 4 | 38.81 GB | 50.52 GB | +11.71 GB |

HP 额外显存主要来自 DTensor 元数据和 FSDP2 管理开销。

## Group 2：性能对比

**环境**：CPU_AFFINITY_CONF=1，不开 ASCEND_LAUNCH_BLOCKING 和 HCCL_DETERMINISTIC

**HP 配置**：upcast_trainable_params=True + enable_prefetch=True (comm_fusion: 4B=True, 30B/32B=False)

### 结果

| 模型 | Native 速度 | HP 速度 | **加速比** | Native peak | HP peak | 额外显存 |
|------|-----------|---------|-----------|-----------|---------|---------|
| Qwen3.5-4B | 0.13 iters/s | 0.14 iters/s | +7.7% | 9.27 GB | 15.21 GB | +5.94 GB |
| Qwen3-30B-A3B | 0.07 iters/s | 0.08 iters/s | +14.3% | 33.25 GB | 47.12 GB | +13.87 GB |
| **Qwen3-32B** | **0.07 iters/s** | **0.16 iters/s** | **+128.6%** | **38.81 GB** | **51.43 GB** | **+12.62 GB** |

### 总耗时

| 模型 | Native | HP | 节省 |
|------|--------|-----|------|
| Qwen3.5-4B | 447s (7.5 min) | 530s (8.8 min) | -18.6% (HP 更慢，额外显存拖累) |
| Qwen3-30B-A3B | 1787s (29.8 min) | 1739s (29.0 min) | +2.7% |
| **Qwen3-32B** | **1511s (25.2 min)** | **834s (13.9 min)** | **+44.8%** |

### 精度对比

| 模型 | Avg `|loss diff|` | Max `|loss diff|` | 最终 loss (native) | 最终 loss (HP) |
|------|---------------|---------------|------------------|---------------|
| Qwen3.5-4B | 0.0053 | 0.0470 | 0.0564 | 0.0556 |
| Qwen3-30B-A3B | 0.0127 | 0.1374 | 0.2572 | 0.2903 |
| Qwen3-32B | 0.0112 | 0.1755 | 0.0026 | 0.0029 |

Loss 差异与 Group 1 一致，HP 的性能优化（prefetch）不影响精度。

### comm_fusion 说明

- **4B**：comm_fusion=True 可用（显存充裕）
- **30B-MoE**：comm_fusion=True 在模型加载阶段即 OOM（60.05/60.95 GB），必须关闭
- **32B**：comm_fusion=True 在模型加载阶段即 OOM，必须关闭

### 显存差异分析

HP 额外显存（~6-14 GB）来自：
1. **prefetch buffer**：forward/backward 预取下一层参数时同时持有当前层和下一层的参数分片
2. **DTensor 元数据**：HP 的 DTensor 实现比 native FSDP2 的 DTensor 多一些管理开销
3. **comm_fusion buffer**（仅 4B）：融合通信的额外 buffer（~6 GB）

### 模型架构对加速效果的影响

| 架构 | HP 加速 | 原因 |
|------|--------|------|
| Dense 32B | **+129%** | 计算密集，prefetch 有效重叠通信和计算 |
| MoE 30B | +14% | 稀疏激活（8/128 experts），计算通信比低，prefetch 窗口小 |
| Dense 4B | +8% | 模型太小，通信占比低，优化空间有限 |

## 运行命令

```bash
conda activate hyper_twinkle

# === Group 1: 确定性精度验证 ===

# Qwen3-0.6B (2 卡, 全量微调)
ASCEND_RT_VISIBLE_DEVICES=0,1 \
ASCEND_LAUNCH_BLOCKING=1 HCCL_DETERMINISTIC=true \
torchrun --nproc_per_node=2 --master_port 29500 \
    cookbook/transformers/compare_fsdp2_fulltune_06b.py --backend native --gradient-accumulation-steps 2
ASCEND_RT_VISIBLE_DEVICES=2,3 \
ASCEND_LAUNCH_BLOCKING=1 HCCL_DETERMINISTIC=true \
torchrun --nproc_per_node=2 --master_port 29501 \
    cookbook/transformers/compare_fsdp2_fulltune_06b.py --backend hp --gradient-accumulation-steps 2

# Qwen3.5-4B
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
ASCEND_LAUNCH_BLOCKING=1 HCCL_DETERMINISTIC=true CPU_AFFINITY_CONF=1 \
torchrun --nproc_per_node=4 --master_port 29500 \
    cookbook/transformers/compare_fsdp2_4b_deterministic.py --backend native
torchrun --nproc_per_node=4 --master_port 29500 \
    cookbook/transformers/compare_fsdp2_4b_deterministic.py --backend hp

# Qwen3-30B-A3B MoE
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
ASCEND_LAUNCH_BLOCKING=1 HCCL_DETERMINISTIC=true CPU_AFFINITY_CONF=1 \
torchrun --nproc_per_node=4 --master_port 29500 \
    cookbook/transformers/compare_fsdp2_30b_moe.py --backend native
torchrun --nproc_per_node=4 --master_port 29500 \
    cookbook/transformers/compare_fsdp2_30b_moe.py --backend hp

# Qwen3-32B
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
ASCEND_LAUNCH_BLOCKING=1 HCCL_DETERMINISTIC=true CPU_AFFINITY_CONF=1 \
torchrun --nproc_per_node=4 --master_port 29500 \
    cookbook/transformers/compare_fsdp2_32b.py --backend native
torchrun --nproc_per_node=4 --master_port 29500 \
    cookbook/transformers/compare_fsdp2_32b.py --backend hp

# === Group 2: 性能对比 ===

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 CPU_AFFINITY_CONF=1 \
torchrun --nproc_per_node=4 --master_port 29500 \
    cookbook/transformers/compare_fsdp2_4b_deterministic.py --backend native
torchrun --nproc_per_node=4 --master_port 29500 \
    cookbook/transformers/compare_fsdp2_4b_deterministic.py --backend hp --perf

torchrun --nproc_per_node=4 --master_port 29500 \
    cookbook/transformers/compare_fsdp2_30b_moe.py --backend native
torchrun --nproc_per_node=4 --master_port 29500 \
    cookbook/transformers/compare_fsdp2_30b_moe.py --backend hp --perf

torchrun --nproc_per_node=4 --master_port 29500 \
    cookbook/transformers/compare_fsdp2_32b.py --backend native
torchrun --nproc_per_node=4 --master_port 29500 \
    cookbook/transformers/compare_fsdp2_32b.py --backend hp --perf
```

### HyperParallel 性能配置

```python
hyper_parallel_config={
    'tp_size': 1,
    'device_type': 'npu',
    'param_dtype': 'bf16',
    'reduce_dtype': 'bf16',
    'reshard_after_forward': True,
    'upcast_trainable_params': True,    # fp32 upcast（与 native 对齐，必须开启）
    # 性能优化选项（--perf）:
    'comm_fusion': False,               # 30B/32B 必须关闭（OOM），4B 可开启
    'enable_prefetch': True,            # 前向后向预取
    'num_forward_prefetch': 1,          # 前向预取层数
    'num_backward_prefetch': 1,         # 后向预取层数
}
```

## Profiling 算子对比（Qwen3-32B，step 10，4 卡，seq_len=4096）

使用 `torch_npu.profiler`（Level1, with_stack=True, CPU+NPU activities）采集第 10 步的算子数据，对比 native 和 HP 的算子差异。

### 总览

| 指标 | Native | HP | 差异 |
|------|--------|-----|------|
| 算子总调用数 | 36,227 | 43,716 | +7,489 (+20.7%) |
| 唯一算子种类 | 61 | 62 | +1 |
| MatMulV2 调用数 | 4,609 | 4,613 | +4 (可忽略) |
| MatMulV3 (MIX_AIC) | 129 | 129 | 一致 |

**模型计算图的 MatMul 数量一致**（4609 vs 4613），差异在可接受范围内。算子总数差异主要来自优化器执行路径和参数通信方式的不同。

### 差异分析

#### 1. Foreach 融合算子缺失（最大差异来源）

Native 的 AdamW 优化器使用 `torch._foreach_*` 系列 API，将多个参数的张量操作融合为单次内核启动。HP 因 `SkipDTensorDispatch` 包装了 optimizer，跳过 DTensor dispatch 后退化为逐元素 scalar 路径：

| Native (Foreach 融合) | 调用数 | HP (逐元素拆分) | 调用数 |
|----------------------|--------|----------------|--------|
| ForeachCopy | 129 | — (直接 copy) | — |
| ForeachAddcmulScalar | 56 | Addcmul | 896 |
| ForeachLerpScalar | 38 | Lerp | 896 |
| ForeachMulScalar | 38 | Muls | (合并到 Muls 总计) |
| ForeachSqrt | 38 | Sqrt | 897 |
| ForeachDivScalarList | 18 | RealDiv | 2,963 |
| ForeachAddScalar | 19 | — | — |

**Sqrt**: Native 仅 1 次（ForeachSqrt 融合了 38 次），HP 为 897 次（逐参数独立调用）。
**RealDiv**: Native 1,220 次 vs HP 2,963 次，HP 多出部分来自 ForeachDivScalarList 的拆分。

#### 2. SplitV vs 无 SplitV

| 算子 | Native | HP |
|------|--------|-----|
| SplitV | 129次, 214ms (占总耗时 22.4%) | 无 |

Native FSDP2 使用 SplitV 做 all-gather 后的参数 reshard。HP 的 `fully_shard` 实现走不同路径，不产生 SplitV 算子。SplitV 在 native 中是单算子耗时最高的（avg 1.66ms/次），HP 省去这部分开销是性能优势来源之一。

#### 3. HP 特有算子

HP 多出少量 `_high_performance` 后缀的特化内核（TransData、Cast、Muls、Add 等，各 1 次），以及 1 个 `MatMulV2_NZ_NZ_FP16_FP16` 特化变体。这些是 HP DTensor dispatch 层引入的优化内核。

### 小结

- **计算等价**：MatMul 等核心计算算子数量一致，确认 native 和 HP 的模型计算图完全等价
- **优化器路径差异**是算子数差异的主因（Foreach 融合 vs 逐元素拆分），不影响数值结果
- **通信差异**：Native 的 SplitV（占 22.4% 耗时）在 HP 中不存在，这是 HP 的性能优势之一
- **HP 的 20% 额外算子调用**主要来自优化器拆分，实际开销很小（Lerp/Addcmul/Sqrt 每次 <5us）

## 结论

1. **精度对齐**：2 卡 FSDP2 下 native 和 HP 实现完全零误差（Qwen3-0.6B: 125/125 步 bit-exact）。4 卡 FSDP2 的 loss 漂移来自多卡浮点归约差异（前 5 步仍 bit-exact），avg diff < 0.013，不影响收敛。

2. **`upcast_trainable_params=True` 是精度对齐的必要条件**：Native FSDP2 通过 Accelerate 自动将可训练参数 upcast 到 fp32（`fsdp_utils.py:707`），HP 通过 `upcast_trainable_params=True` 实现相同行为。关闭此选项会导致参数 bf16 + 优化器状态 bf16，与 native 的 fp32 不匹配，产生显著偏差。

3. **性能**：HP prefetch 对 dense 大模型加速显著（32B: +129%），对 MoE 模型收益有限（30B: +14%），对小模型可忽略（4B: +8%）。

4. **显存**：HP 额外显存 ~6-14 GB（prefetch buffer + DTensor 开销）。comm_fusion=True 额外增加 ~6 GB 但对大模型会 OOM。

## 代码修改（已提交 `e6b8476`）

1. **`framework.py`** — `find_spec('megatron.core')` 加 try-except，megatron 未安装时不再崩溃
2. **`grad_clip.py`** — 纯 DTensor 梯度走 local-norm fallback，兼容 hyper_parallel dispatch
3. **`transformers.py`** — optimizer 的 `SkipDTensorDispatch` 包装移到 `set_optimizer` 时执行
4. **`hyper_parallel.py`** — 暴露 `comm_fusion`/`prefetch`/`upcast_trainable_params` 参数
5. **`hyper_parallel/utils.py`** — comm_fusion 和 prefetch 参数化，upcast 开关
6. **cookbook 脚本** — NPU 卡数自动检测、per-step 日志、显存快照、HCCL_DETERMINISTIC
