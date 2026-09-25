# Task4：多请求调度、异构 PD 与 Serving

- 学习者：GitHub `zcloveyou2333` / 微信群 `zc_loveyou`
- 打卡任务：[llm-algo-leetcode 推理优化 | 202609 Task4 · 多请求调度、异构 PD 与 Serving](https://github.com/datawhalechina/llm-algo-leetcode/issues/174)
- 官方截止时间：2026-09-26 12:00
- 打卡范围：完成 4.1；4.2、4.3 作为后续深入实验
- 运行环境：macOS，Python 3.10.21，PyTorch 2.14.0，CPU
- 教程来源：[Datawhale 社区](https://github.com/datawhalechina)的 [llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode)

## 1. 三层调度分别在决定什么

三类调度不是互相替代，而是从“选谁”逐步走向“资源能否容纳”和“这轮怎样共同执行”。

| 层次 | 主要决策单位 | 关键状态 | 输出 |
| --- | --- | --- | --- |
| Decode 调度 | ready 请求或 sequence | 阶段、生成进度、优先级、等待时间、cache 命中 | 下一步推进哪些请求 |
| KV Cache 调度 | 请求的 cache 需求、可复用 cache entry 或物理 block | 已用/空闲容量、命中热度、引用与锁定、可驱逐性 | 请求是否可接纳，以及复用、分配或驱逐哪些块 |
| Prefill/Decode 调度 | Prefill suffix/chunk、Decode step 和逻辑资源池 | 两类队列、token budget、batch 容量、池利用率、handoff 状态 | 本轮动态 batch、chunk 顺序及是否跨池交接 |

衔接顺序可以写成：

```text
ready 请求选择
  -> 估算本轮新增 KV 需求并做容量准入
  -> 组成迭代级动态 batch（Decode step + Prefill chunk）
  -> 执行并更新生成进度、KV 占用、等待时间和池状态
  -> 下一轮重新决策
```

因此请求级调度只回答“谁更应该运行”，不能保证物理上放得下；KV 调度回答“现有容量是否允许它运行”；Prefill/Decode 调度再把已准入工作组织成可执行的迭代，并在 chunk 或 handoff 边界重新选择。

我补全并执行了 [Part 02 · 36](../../../02_PyTorch_Algorithms/36_Decode_Scheduling.ipynb)。实现用 `RequestState` 保存生命周期，以阶段、cache 命中、业务优先级、总长度和稳定 ID 组成排序键，并在每次选择后更新生成进度及未选请求的等待步数。

![Decode 调度 CPU 测试通过](./assets/task4-decode-scheduling-tests.png)

## 2. 为什么长 Prefill 会干扰 Decode

Prefill 通常一次处理很多 token，计算量和显存带宽需求都明显高于单步 Decode。如果两者共用 GPU 执行队列、batch token budget、Attention kernel 和显存带宽，一个很长的 Prefill 会连续占住执行窗口。已经进入 Decode 的请求即使只差下一个 token，也必须等待，TPOT 抖动和 P95/P99 延迟随之上升。

Chunked Prefill 只把 Prefix Cache 未命中的 suffix 切成较小 chunk。每完成一个 chunk，调度器就获得一个 yield point，可以先插入等待中的 Decode，再继续后续 Prefill。它缩短了单次不可重排的占用时间，但不会免费消除工作量。

代价主要有：

1. chunk 越小，调度次数、kernel launch、元数据维护和同步开销越多；
2. Prefill 的连续计算被打断，算子效率可能下降，长请求自己的 TTFT 可能变差；
3. chunk、Decode 与 token budget 的组合需要调参，过度偏向 Decode 也可能饿死 Prefill；
4. 若进一步采用 PD 分池，还会增加 KV 状态交接、链路传输、失败恢复与容量配平成本。

我补全并执行了 [Part 02 · 38](../../../02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling.ipynb)。测试覆盖请求压力分类、suffix 有序分块、互斥逻辑分池，以及同时使用吞吐、P95 和 handoff 预算做保留决策。

![Prefill/Decode 调度 CPU 测试通过](./assets/task4-prefill-decode-scheduling-tests.png)

## 3. KV Cache 紧张时如何判断是否准入

请求长度只是静态输入之一。调度器至少还应结合以下状态：

- **当前物理容量**：空闲 block 数、保留水位、尾块浪费，以及即将释放但尚未回收的容量；
- **增量需求**：Prefix Cache 实际命中多少 token、未命中 suffix 需要多少新 block，以及 Decode 继续增长的安全余量；
- **缓存生命周期**：block 是否被活动请求引用或锁定、引用计数、是否正在 handoff/offload、能否安全驱逐；
- **复用与驱逐价值**：命中次数、最近访问、entry 大小、重算成本和驱逐后对其他请求的影响；
- **请求运行状态**：Prefill/Decode 阶段、已生成长度、可否抢占、抢占后的重算或恢复成本；
- **服务目标**：等待时间、业务优先级、deadline、TPOT/TTFT 预算与公平性；
- **提交安全性**：准入与 block 分配应能原子提交；中途容量不足时必须回滚，不能留下半分配状态。

一个实用判断不是“总长度小就接纳”，而是比较：

```text
可立即复用的 KV + 可安全分配的空闲容量 + 可接受代价下可回收的容量
是否覆盖本轮增量 KV 需求和安全余量
```

我补全并执行了 [Part 02 · 37](../../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb)。实现使用命中次数、访问新鲜度和容量成本计算保留分数，通过带懒删除的优先级堆避免旧状态误驱逐新 entry，并持续检查容量账本守恒。

![KV Cache 调度 CPU 测试通过](./assets/task4-kv-cache-scheduling-tests.png)

## 4. 复现与证据边界

```bash
conda run --no-capture-output -n llm_algo \
  python tools/test_notebook_answers.py \
  02_PyTorch_Algorithms/36_Decode_Scheduling.ipynb --mode both

conda run --no-capture-output -n llm_algo \
  python tools/test_notebook_answers.py \
  02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb --mode both

conda run --no-capture-output -n llm_algo \
  python tools/test_notebook_answers.py \
  02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling.ipynb --mode both
```

三份 Notebook 的题目区与答案区测试均通过，并已在 `llm_algo` 环境中从头到尾执行。实际输出分别确认：请求状态/排序/推进/保护通过，cache 输入/评分/堆状态/容量守恒通过，以及分类/分块/分池/决策通过。

本机没有 CUDA 环境，所以三个可选 GPU/backend 单元保持关闭：`RUN_GPU_MICROBENCH=False`、`RUN_GPU_BLOCK_PROBE=False`、`RUN_PD_GPU_PREFLIGHT=False`。本次证据只证明 CPU 机制模拟和状态不变量正确，不能据此声称真实 Serving 的 TTFT、TPOT、吞吐、峰值显存或 P99 得到改善。

## 后续深入实验清单

1. 把请求排序、Paged Block 准入和 Chunked Prefill 合成同一个迭代级模拟器，观察容量压力如何改变完成顺序和公平性。
2. 扫描 `chunk_size` 与 Decode token budget，记录 TTFT、TPOT、吞吐、P95/P99 和调度开销的 Pareto 前沿。
3. 为 PD handoff 建立状态机：Prefill 完成、KV 冻结、传输、校验、Decode 接管、失败回滚与超时重试。
4. 对异构 PD 路由同时加入 GPU 算力、显存余量、互联带宽、KV 传输量、队列长度、模型兼容性和 SLA。
5. 在真实 vLLM/SGLang GPU backend 上使用相同 workload，对比共享池、Chunked Prefill 和 PD 分离，并保留环境、warmup、并发与失败证据。
