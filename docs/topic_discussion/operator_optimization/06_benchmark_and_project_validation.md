# 06. Benchmark 与项目验证

## 页面目标

本页把局部 kernel 优化收束成可复查项目：参考实现 → 候选 kernel → 数值对齐 → 固定 workload → microbenchmark → 端到端 benchmark → 决策。

![算子优化证据流：从候选实现到项目决策](../../public/topic_discussion/operator_optimization/operator_optimization_evidence_flow.svg)

## 最小验证表

| 阶段 | 必须固定或记录 | 通过条件 |
|:---|:---|:---|
| 语义对齐 | 输入、输出、shape、dtype、误差阈值 | 输出满足契约 |
| kernel 测试 | GPU、编译选项、warmup、迭代次数 | 结果可重复 |
| 性能比较 | baseline、candidate、shape、workload | 指标在同口径下比较 |
| 端到端验证 | Block、模型、服务或任务 | 局部收益传递到目标路径 |
| 结论记录 | 证据等级、失败条件、适用范围 | 他人可以复查 |

## 实验记录模板

每次比较只改变一个主要因素；如果同时改变 dtype、tile、fusion 和 GPU，就无法解释收益来源。下面的表格可以直接复制到项目报告中。

| 字段 | baseline | candidate | 记录要求 |
|:---|:---|:---|:---|
| 算子 / Block | 参考实现名称 | 候选 kernel 名称 | 说明是否包含 fusion |
| 输入条件 | shape、dtype、layout | 相同 shape、dtype、layout | 先固定共同条件 |
| 执行环境 | GPU、驱动、PyTorch、CUDA | 相同环境 | 更换环境需重新测量 |
| 编译配置 | 编译目标、编译选项 | Triton / CUDA 配置 | 记录编译时间和缓存状态 |
| 计时设置 | warmup、repeats、同步方式 | 相同设置 | CUDA 计时前后同步 |
| 正确性 | 误差指标和阈值 | 与 baseline 对照 | 不通过时停止性能结论 |
| 性能结果 | kernel time、显存 | kernel time、显存 | 标注单位和统计方式 |
| 系统结果 | Block / workload 指标 | Block / workload 指标 | 记录延迟、吞吐和适用范围 |

## 项目报告至少回答什么

- 优化改变了哪一段语义或数据路径？
- 哪些 shape、dtype 和 GPU 被验证？
- kernel 时间、显存、编译成本和端到端指标如何变化？
- 结果是否可重复，哪些条件下会失效？
- 最终结论是 `accept`、`tune` 还是 `reject`？

如果候选 kernel 通过了单独的 microbenchmark，还要确认上层调用路径是否真的使用它。报告应注明调用方式是 PyTorch eager、`torch.compile` / Inductor、Triton wrapper、CUDA extension 还是其他 backend，并说明编译、缓存和 fallback 行为。

| 集成层 | 需要验证 | 失败时如何记录 |
|:---|:---|:---|
| 直接调用 | candidate kernel 是否被实际调用 | 记录调用入口和 kernel 名称 |
| 编译路径 | Inductor / JIT 是否保留或替换实现 | 记录生成代码、fallback 或 graph break |
| Block 集成 | 模型组件输出和布局是否仍对齐 | 分别记录组件与 Block 结果 |
| 端到端路径 | workload 是否得到目标收益 | 不把局部 kernel 时间写成系统收益 |

## 证据等级

| 等级 | 说明 | 可以支持的结论 |
|:---|:---|:---|
| `cpu_correctness` | 参考实现与边界测试通过 | 语义和接口正确 |
| `gpu_microbenchmark` | 固定 shape 下的 GPU kernel 对照 | 局部 kernel 成本变化 |
| `workload_benchmark` | 固定模型或 Block 的端到端对照 | 当前 workload 的收益 |
| `project_decision` | 重复运行、质量门槛和适用边界齐全 | 有条件的采用建议 |

没有匹配 workload 或真实 trace 时，只能记录为优化假设，不能写成通用结论。GPU 实验与环境说明应沿用项目页的运行环境和配置记录方式。

## 本页出口

完成本页后，应能提交一份包含正确性、性能、环境、失败条件和决策标签的最小报告，而不是只提交一个“更快”的数字。
