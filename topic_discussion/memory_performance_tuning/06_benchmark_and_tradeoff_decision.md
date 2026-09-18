# 06. Benchmark and Trade-off Decision | Benchmark 与取舍决策

## 页面目标

本节对应 **Task3：训练侧 Benchmark 与预算决策**。这里把训练侧项目的机制、策略和证据收束到同一个判断框架中，回答显存省了多少、时间和质量付出了什么代价、是否扩大了 batch，以及最终是否值得保留。Task6 的分布式扩展见 [07 分布式显存与系统扩展](./07_distributed_memory_and_system_extension.md)。

## Task0–6 知识覆盖检查

路线中的 Task 不只是项目编号，还要把机制、策略和证据出口串起来。当前覆盖关系如下：

| Task | 核心问题 | 主要机制入口 | 验证出口 | 当前覆盖判断 |
|:---|:---|:---|:---|:---|
| Task0 | 状态为什么产生、驻留并释放？ | `07 → 18 → 17`；Autograd 生命周期、局部梯度、Attention backward、saved tensors 和 activation | CPU 机制检查 | 已覆盖；后续可补充理论账本观察 |
| Task1 | 状态如何进入容量和峰值账本？ | Part01 `01`、`02`、`03`、`06`；`04 / 12 / 14` 作为共享支撑 | CPU 账本；GPU peak / reserved / OOM | 核心完整；架构内容作为扩展，不作为共同前置 |
| Task2 | 如何用微步、重算或搬运换显存？ | `12`、`19`、`42`；effective batch、checkpoint、offload | GPU 策略比较 | 已覆盖；需要统一策略代价表 |
| Task3 | 如何比较训练策略并做预算决策？ | Part00 `20`、Part01 `13`；测量口径和证据等级 | `73 → 76 → 75 → 74` | 已形成项目链；机制入口已补齐 |
| Task4 | KV Cache 如何增长、组织和复用？ | Part01 `11`；`22`、`34`；`24 / 71` 为架构扩展 | `66 / 69 / 71` backend 实验 | 核心完整；服务调度不放入本专题主线 |
| Task5 | 量化是否真正改变容量边界？ | `21`、`25`、`40`、`41`；对象、时机和 backend | `67` 真实部署对照 | 已覆盖；需保持 KV Cache 量化归入本 Task |
| Task6 | 单卡放不下时如何分摊状态并解释通信代价？ | `27–29`；`79–81`；Profiling 为共享扩展 | 多 GPU / 分布式实验；复用 `74` trace | 分布式主线完整；不把 Profiling 当共同前置 |

这张表用于检查路线是否缺少知识领域，不替代各 Notebook 的实验步骤。若某一行只有项目而没有机制入口，先补正文或共享前置，再进入项目实验。

## 核心机制

显存 benchmark 不是单独比较一个峰值数字，而是建立可比的 baseline / candidate 对照。至少固定模型、revision、dtype、workload、batch、seq_len、warmup、iters、seed、硬件和软件版本；候选只改变一个主要策略。

| 项目 | 主要职责 | 必须记录的证据 | 输出 |
|:---|:---|:---|:---|
| 73 | 建立训练 baseline | step time、吞吐、peak / reserved、loss、OOM | 固定训练口径 |
| 76 | 比较 checkpoint、offload、hybrid | 显存、吞吐、质量、状态和策略差异 | 候选对比 |
| 75 | 改变显存和吞吐门槛 | 可行集合、最优候选、阈值敏感性 | `accept / tune / reject` |
| 74 | 解释候选方案为什么变快或变慢 | profiler trace、重算、搬运、kernel 和端到端时间 | 证据收口 |

![Benchmark：从对照实验到资源决策](../../docs/public/topic_discussion/memory_performance_tuning/benchmark_tradeoff_decision.svg)

| 验证分支 | 主要项目 | 主要指标 | 需要回答的问题 |
|:---|:---|:---|:---|
| 训练显存 | 73、76、75 | peak / reserved、step time、吞吐、loss、OOM | 当前策略是否值得保留？ |
| 推理缓存 | 66、69、71 | cache 容量、并发、命中率、TTFT、TPOT | 缓存组织是否扩大了可用上下文或并发？ |
| 量化部署 | 67 | 格式、kernel、显存、吞吐、质量 | 压缩是否真正改变了部署边界？ |
| 分布式显存 | 79、80、81 | 单卡显存、通信时间、扩展效率、稳定性 | 分摊显存是否值得通信和系统复杂度？ |
| Profiling 收口 | 74 | trace、重算、搬运、kernel、端到端时间 | 指标变化是否有可解释的瓶颈证据？ |

## 判断与验证

训练侧按 `73 → 76 → 75 → 74` 建立主证据链：

1. `73` 先固定环境和 workload，建立 baseline；BF16、长序列、LoRA / QLoRA 属于扩展 workload，应单独记录。
2. `76` 在同一 workload 下比较 `baseline`、`checkpoint`、`offload` 和 `hybrid`，同时检查质量门槛和 OOM 状态。
3. `75` 直接读取 76 的 JSON，改变显存上限、吞吐下限和质量门槛；不重新训练，观察决策是否对阈值敏感。
4. `74` 使用真实 profiler trace 检查显存变化是否对应 activation、重算、搬运或 kernel；没有 trace 时只能报告证据缺口。

其他分支沿相同原则独立验证：66 / 69 / 71 固定请求 workload 比较缓存策略，67 使用真实量化 artifact 与 FP16 / BF16 baseline 对照，79–81 固定模型切分和多卡环境记录显存分摊与通信。不同分支的数字不能直接合并，但可以使用同一套“对象、指标、约束、证据、决策”字段。

证据等级也要随实验类型区分：

| 证据等级 | 能确认什么 | 不能确认什么 |
|:---|:---|:---|
| CPU 机制检查 | 公式、shape、生命周期、指标聚合和决策逻辑 | GPU 峰值、带宽、kernel、OOM 和 backend 行为 |
| 单 GPU 对照 | 固定 workload 下的峰值、吞吐、OOM 和策略代价 | 其他 GPU、其他模型或所有服务负载都同样成立 |
| 真实 backend / 多 GPU | cache、量化格式、通信和服务级结果 | 超出当前 workload 和环境的普遍结论 |
| Trace 收口 | 时间和显存变化的具体来源 | 没有匹配 workload 时的因果归因 |

判定标准：

- `accept`：显存、性能、质量和稳定性同时满足约束；
- `tune`：方向可行，但收益不稳定、阈值敏感或证据不足；
- `reject`：当前 workload 下收益不足、副作用过大或质量不达标。

历史报告中的显存节省、吞吐变化和 `accept / tune / reject` 只适用于报告记录的模型、硬件、dtype 和 workload。阅读新报告时，先检查 baseline / candidate 是否同口径，再判断候选是否满足当前预算；如果 74 缺少与候选 workload 匹配的 trace，只能保留为证据缺口，不能写成完整 profiling 结论。

CPU 可以验证指标聚合、预算筛选、阈值变化和决策逻辑；GPU 才能确认真实峰值、吞吐、OOM 边界和策略代价。项目命令、结果文件和数据登记要求见[73–76 显存优化项目验证清单](../../docs/verification/memory_projects.md)。
