# 01. VRAM Ledger and Metrics | 显存账本与指标口径

## 页面目标

本节对应 **Task1：dtype、模型规模、硬件与显存账本**。这里建立一套可以跨训练和推理复用的显存账本，回答显存由哪些对象构成、峰值可能出现在哪个阶段、理论估算和 GPU 实测分别能说明什么。

## 核心机制

显存问题不是单一的“参数太大”。参数、梯度、optimizer state、activation、KV Cache 以及框架 buffer 的生命周期不同，峰值也不一定同时出现。因此，账本需要同时记录对象、变化因素和证据出口。

账本先记录“对象—规模—dtype—生命周期—证据”五个字段。以 AdamW 为例，参数本身只是模型状态的一部分，优化器还会为可训练参数维护额外状态；不能用“参数字节数”代替完整训练账本。

| 显存对象 | 主要变化因素 | 账本中先记录什么 | 需要什么证据 |
|:---|:---|:---|:---|
| 参数、梯度 | 参数量、dtype、训练 / 推理模式 | 理论字节数和是否常驻 | GPU 是否能加载、峰值显存 |
| optimizer state | 优化器类型、参数量、更新状态 | 单独估算常驻成本 | 训练显存和 step 稳定性 |
| activation | batch、sequence length、保存策略 | 产生和释放的生命周期 | checkpoint 前后峰值、重算时间 |
| KV Cache | 层数、KV heads、上下文、并发 | 每请求增长和驻留方式 | cache 容量、并发、命中率 |
| workspace / fragment | kernel、allocator、backend | 作为剩余预算和运行时开销 | `reserved memory`、trace 或 OOM |

这些对象的峰值不一定同时出现，所以理论账本是容量估算或诊断近似，不能把各阶段峰值机械相加。

![显存账本：对象、生命周期与证据](../../docs/public/topic_discussion/memory_performance_tuning/memory_ledger.svg)

| 指标 | 它回答什么问题 | 适合怎样使用 |
|:---|:---|:---|
| `peak memory` | 运行中最高占用是否超过预算 | 判断是否装得下、是否接近 OOM |
| `reserved memory` | allocator 已保留但当前未必使用多少 | 观察碎片、buffer 和缓存行为 |
| `memory delta` | 候选方案减少了多少峰值 | 比较 baseline / candidate |
| `step time`、`throughput`、`latency` | 节省显存付出了多少时间 | 判断优化是否值得 |

一个可复查的最小账本至少包含以下内容：

| 字段 | 示例问题 | 结果类型 |
|:---|:---|:---|
| 对象与生命周期 | 这是参数、activation 还是 KV Cache？在哪个阶段驻留？ | 机制说明 |
| 规模与 dtype | 有多少元素，每个元素占多少字节？ | 理论估算 |
| 峰值与保留量 | 运行中最高占用是多少，allocator 保留了多少？ | GPU 实测 |
| 代价与质量 | 节省显存后，时间、吞吐或质量变化如何？ | 对照实验 |

理论账本用于提出假设，GPU 结果用于验证假设；两者应保留在同一份报告中，但不能互相替代。

## 判断与验证

遇到显存问题时，按下面顺序把账本转成实验问题：

1. 先区分训练显存和推理显存，再指出主要对象。
2. 把对象放回生命周期：forward、backward、optimizer step、prefill 或 decode。
3. 判断当前目标是“装得下”，还是“装下以后仍有可接受的吞吐和质量”。
4. 用 CPU 验证字节数、对象分类和账本关系；用 GPU 项目验证峰值、reserved、吞吐和 OOM 边界。

Task0 解释计算图和状态生命周期，Task1 的 [01 数据类型](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb)、[02 参数量与 FLOPs](../../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb)、[03 GPU 架构与显存](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb) 和 [06 显存计算与 ZeRO](../../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb) 再把对象放入硬件和状态账本。需要连接 Attention 访存时，补看 [20 FlashAttention Sim](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb)。

后续验证出口按问题选择：训练侧进入 [73 训练性能分析](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb)、[76 策略对比](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb) 和 [74 Profiling 收口](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb)；推理侧进入 [04 推理 Cache 与显存预算](./04_inference_cache_and_memory_budget.md)。

CPU 结果只能说明账本和机制；即使代码运行在带 GPU 的机器上，`device='cpu'` 的结果也不能写成 GPU 峰值、带宽或 OOM 结论。
