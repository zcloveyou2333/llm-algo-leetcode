# 02. Training Memory Pressure | 训练侧显存压力

## 页面目标

本节是 **Task0–2 的训练侧机制入口**：先解释状态为什么在计算图中驻留，再把“训练一上大 batch 就 OOM”拆成可测量的问题，最后为 Task2 的策略比较准备对象和代价语言。

## 核心机制

训练显存的关键在于一次前向和反向需要保留什么。`micro-batch` 决定单次 forward / backward 的 activation 压力；`gradient accumulation` 决定多少个 micro-step 汇总成一次 optimizer update。增加 accumulation 通常不会减少参数、梯度或 optimizer state，也不会继续降低已经固定的单步 activation 峰值。

| 现象或对象 | 需要先理解的关系 | 主要影响因素 |
|:---|:---|:---|
| activation 峰值 | 前向产生的中间结果可能要留到 backward | batch、sequence length、层数、保存策略 |
| gradient | 反向后用于更新参数，通常跨 micro-step 保留到 update | 参数量、dtype、梯度累积方式 |
| optimizer state | 更新参数所需的额外状态，不等于参数本身 | AdamW 等优化器、参数量、dtype |
| effective batch | 多个 micro-step 的样本总量 | micro-batch、accumulation steps |

因此，缩小 batch 只是一个控制旋钮；它可能降低 activation 峰值，却同时改变吞吐和训练节奏。真正的策略选择要先确认主因，再判断是否需要 accumulation、checkpoint、offload、sharding 或量化。

先用下面这张生命周期表建立 Task0–2 共用的观察语言。对象的“大小”只能说明容量压力，只有把产生、驻留和释放阶段放回时间线，才能解释峰值为什么出现在某个位置。

| 对象 | 什么时候产生 | 什么时候驻留 | 什么时候释放或复用 | 首要观察量 |
|:---|:---|:---|:---|:---|
| 参数 | 模型加载时 | 整个训练或推理过程 | 进程结束或模型卸载 | 参数量、dtype、加载峰值 |
| activation | forward 计算过程中 | 等待 backward 使用 | 对应反向节点完成后 | saved tensors、阶段峰值 |
| 梯度 | backward 计算过程中 | 直到 optimizer step 或下一轮清零 | `zero_grad` 或梯度覆盖 | 梯度 dtype、累积方式 |
| optimizer state | 第一次或后续参数更新时 | 训练期间持续驻留 | 优化器释放或参数移除 | state 大小、更新后峰值 |
| 临时 workspace | 算子或 backend 执行时 | 当前 kernel / 阶段 | 算子完成后回收或进入 allocator cache | allocated、reserved、trace |

这张表解释了为什么 Task0 先学习生命周期，Task1 再把对象放入账本，Task2 才选择重算、累积或搬运。推理侧的 KV Cache 也遵循同一语言，但它按请求和 token 增长，在 Task4 单独展开。

![训练显存压力：从输入规模到策略选择](../../public/topic_discussion/memory_performance_tuning/training_pressure_diagnosis.svg)

## 判断与验证

先沿着“对象 → 策略 → 代价 → 证据”读这一节：

在选择策略前，先用一个小 workload 改变一个输入，确认压力对象：

| 观察动作 | 如果变化明显 | 优先怀疑 | 下一步 |
|:---|:---|:---|:---|
| 增加 micro-batch 或 sequence length | 单步峰值随之上升 | activation 或临时张量 | 进入 checkpoint / offload 对照 |
| 保持 effective batch，改变 accumulation steps | 单步峰值变化有限，但步数和时间变化 | micro-step 与吞吐代价 | 进入 12 检查梯度对齐和有效 batch |
| 更换 optimizer 或增加可训练参数 | 更新后常驻显存变化 | optimizer state | 回到 Task1 账本，再决定是否需要参数分摊或参数高效训练 |
| OOM 出现在 forward、backward 或 step 的不同阶段 | 峰值阶段发生变化 | 不同生命周期对象 | 用 73 固定阶段和 workload，再进入 76 |

| 策略 | 主要改变什么 | 没有改变什么 | 代价与验证 |
|:---|:---|:---|:---|
| Gradient Accumulation | 单个 micro-batch 的 activation 峰值 | 参数、梯度、optimizer state 的规模 | 微步数增加；在 12 中检查有效 batch 和梯度对齐 |
| Checkpointing | 需要长期保存的中间 activation | 参数、梯度、optimizer state | backward 重算；在 19 和 76 中比较峰值与 step time |
| Offload | GPU 上驻留的部分 activation 或状态 | 状态总量 | CPU-GPU 搬运和同步；42 先做模型，76 再做真实比较 |

CPU 可以验证张量生命周期、梯度对齐和账本变化；GPU 才能确认 activation 峰值、重算时间、搬运时间、吞吐和 OOM 边界。小规模机制结果不能直接替代 73、76 的固定 workload。

训练侧项目按照 [73 baseline](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md) → [76 策略比较](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md) → [75 预算决策](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.md) → [74 Profiling](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md) 形成证据链：先固定模型、dtype、batch、seq_len、warmup、iters 和 seed，再比较候选，最后检查预算与 profiler 解释是否一致。

本节对应 Task0–2 的训练侧基础：Task0 说明为什么状态会驻留，Task1 说明这些状态如何进入账本，Task2 才讨论如何用微步、重算或搬运换显存。推理 KV Cache 和权重量化属于后续分支，分别进入 [04 推理 Cache 与显存预算](./04_inference_cache_and_memory_budget.md) 和 [05 量化作为显存工具](./05_quantization_as_a_memory_tool.md)。
