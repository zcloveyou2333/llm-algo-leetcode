# 02. Prefill and Attention Kernel | Prefill 与 Attention Kernel

## 页面目标

从一个长 Prompt 请求开始，观察 Prefill 为什么会推高 `TTFT`，再区分 Attention 访存、Chunked Prefill 和 Prefix Cache 分别解决哪类问题。

## 核心机制

Prefill 处理已有 Prompt，Prompt 变长通常会推高 `TTFT`。先把“标准 Attention + 一次完整 Prefill”作为参考行为：不使用 FlashAttention、不分块，也不复用前缀，在固定 Prompt、模型和硬件下记录 `TTFT`、Attention 时间与峰值显存。再区分三类瓶颈：Attention 访存、单次 Prefill 规模，以及重复前缀计算。下图呈现基线与候选机制的关系，表格用于快速分流；FlashAttention 处理访存，Chunked Prefill 处理单次工作量，Prefix Cache 则作为重复前缀的 Prefill 优化入口。

![Prefill 与 Attention 访存](../../docs/public/topic_discussion/inference_optimization/prefill_attention_zh.svg)

| 机制 | 主要改变什么 | 适合解决的问题 |
|:---|:---|:---|
| 标准 Attention + 完整 Prefill（基线） | 一次处理完整 Prompt，不分块、不复用前缀 | 建立 `TTFT`、Attention 时间和峰值显存参照 |
| FlashAttention | Attention 的访存路径和中间结果写回 | Attention 计算受 HBM 读写拖慢 |
| Chunked Prefill | 长 Prompt 的处理和调度方式 | 单次 Prefill 过大、影响其他请求 |
| Prefix Cache | 重复前缀是否重新计算 | 多请求共享相同前缀，减少重复 Prefill |

三类机制的验证入口不同：FlashAttention 先验证 Attention 工作集和访存行为，Chunked Prefill 先验证长 Prompt 是否影响其他请求，Prefix Cache 先验证前缀是否命中。这里关注 Prefix Cache 对 Prefill 的影响；命中后的状态如何分页、失效、驱逐和治理，在正文 04 中继续展开。它们不能只用 `TTFT` 一个数字判断，因为一个机制可能降低单请求时间，另一个机制可能主要改善并发下的尾部延迟。

| 候选机制 | 最小对照 | 关键证据 | 需要同时观察 |
|:---|:---|:---|:---|
| FlashAttention | 标准 Attention vs 分块实现 | Attention 时间、工作集或显存读写变化 | TTFT、peak memory、输出一致性 |
| Chunked Prefill | 完整 Prefill vs 分块 Prefill | Prefill 被拆分后的调度行为 | TTFT、TPOT、P99、Decode 抖动 |
| Prefix Cache | 不复用前缀 vs 命中前缀 | `hit_len`、`reused_tokens` | TTFT、命中率、Cache 占用 |

Prefill 的时间还可以拆成三类成本：矩阵计算成本、HBM 与片上存储之间的数据搬运成本，以及 kernel launch、同步和形状处理带来的调度成本。Roofline 可以帮助判断算子更接近计算上限还是带宽上限，但它只提供定位假设；最终仍要用 kernel 时间、显存带宽、利用率和端到端 TTFT 验证。

| 成本来源 | 常见信号 | 优先检查 | 可能的动作 |
|:---|:---|:---|:---|
| 计算受限 | 算术强度较高，计算单元利用率高 | FLOPs、Tensor Core 路径、矩阵形状 | 调整 dtype、shape、kernel 或融合 |
| 访存受限 | 带宽接近上限，计算单元未充分利用 | HBM 流量、工作集、读写次数 | 分块、片上复用、FlashAttention |
| 启动/同步受限 | 单次 kernel 很短但 kernel 数量多 | launch 次数、空隙和同步点 | 融合、批处理、CUDA Graph 或编译优化 |
| 形状受限 | 长度变化导致路径和性能抖动 | padding、bucket、动态 shape | 分桶、稳定 shape、重新组织 workload |

FlashAttention 的演进可以按“减少中间状态写回、提高并行效率、匹配硬件路径”理解。它们保持 Attention 的计算语义，主要改变 kernel 的数据流和执行方式；版本差异适合通过 kernel 时间、工作集、带宽和端到端 `TTFT` 验证，而不是只比较名称。

| 实现方式 | 主要变化 | 重点观察 |
|:---|:---|:---|
| 标准 Attention | 显式产生较大的中间矩阵 | 峰值显存、HBM 写回、工作集 |
| FlashAttention | 分块计算、在线 Softmax、片上数据复用 | Attention 时间、带宽、TTFT |
| FlashAttention 2 | 改进并行划分和线程块工作分配 | kernel 时间、GPU 利用率、形状敏感性 |
| FlashAttention 3/4 路径 | 面向更新硬件的异步与低精度执行 | Tensor Core 路径、dtype、实际 GPU 证据 |

参考入口：论文 [FlashAttention](https://arxiv.org/abs/2205.14135)；开源实现 [FlashAttention](https://github.com/Dao-AILab/flash-attention)。

## 判断框架

本节承接 `01` 的请求阶段和指标：先用 [20 FlashAttention Sim](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb) 理解 Attention 访存，再用 [34 Prefix Caching and Chunked Prefill](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.ipynb) 观察长 Prompt 的分块与前缀复用，最后在 [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) 中固定 workload，检查这些机制是否真的改善了请求表现。阅读下表时，先找最接近当前现象的一行，再决定下一步学习或实验。

| 观察到的现象 | 优先判断 | 下一步 |
|:---|:---|:---|
| Prompt 变长时 `TTFT` 持续升高 | Prefill 或 Attention 访存受限 | 检查 FlashAttention 和硬件支持 |
| 长 Prompt 阻塞其他请求 | 单次 Prefill 影响调度 | 检查 Chunked Prefill |
| 多请求包含相同前缀 | 重复计算占主要成本 | 检查 Prefix Cache |
| `TTFT` 高但 Prefill 占比不高 | 排队、batch 组装或服务调度 | 进入 `04` / `06` |
