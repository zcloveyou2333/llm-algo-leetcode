# 81. Distributed Inference Project | 分布式推理逻辑验证
**难度：** Hard | **环境：** CPU-first | **标签：** `并行通信`, `分布式推理`, `推理服务` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/81_Distributed_Inference_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---
## 本节导读

本节先用单卡逻辑模拟验证分布式 serving 是否值得迁移。你需要固定请求 workload、虚拟集群和路由规则，比较 baseline 与候选方案的 makespan、负载均衡、通信成本和总吞吐。最终输出迁移建议，并明确还需要哪些真实多卡数据才能进入部署验证。
**层级定位：** 本项目连接 L4 与 L5：L4 关注分布式推理实例、切分和路由执行，L5 关注副本、资源编排和服务迁移治理；当前 Notebook 只完成逻辑验证，不能替代真实多卡通信和生产平台验证。

**关键词：** `distributed inference`, `routing`, `load balance`, `communication cost`, `migration decision`

---
## 前置阅读

**导语：** 先把基础推理对比、serving 调度、并行 benchmark 和 `2.9` 的分布式主线理顺，再进入这个项目；本节默认你已经知道单机 serving 和并行策略的基本口径，重点转向是否值得迁移到分布式部署。
- [66. Inference Performance Comparison | 推理性能对比实验](./66_Inference_Performance_Comparison.md)
- [70. Serving Scheduler Benchmark | 推理服务调度基准](./70_Serving_Scheduler_Benchmark.md)
- [79. Distributed Parallel Benchmark | 分布式并行基准项目](./79_Distributed_Parallel_Benchmark.md)

## 相关项目入口

**导语：** 完成分布式推理逻辑验证后，用 80 补充 MoE 专家并行对照；如果准备进入真实部署，先用 74 的 profiling 闭环检查迁移收益。
- [80. MoE Expert Parallel Benchmark | MoE 专家并行基准](./80_MoE_Expert_Parallel_Benchmark.md)
- [74. Profiling-Driven End-to-End Optimization | profiling 驱动的端到端优化项目](./74_Profiling_Driven_End_to_End_Optimization.md)

---
### Step 1: 定义 distributed inference 项目目标

- 固定请求集合、虚拟副本数、TP / PP 逻辑度和路由策略。
- 每个请求只记录 `prompt_tokens` 和 `generate_tokens`。
- 在单机环境里先把路由、prefill、decode 和通信逻辑说清楚。
### Step 2: baseline 和迁移口径先要合法

- 分布式推理验证不能脱离基线 workload 讨论。
- 如果 baseline 的请求集合、容量约束或路由规则本身不稳定，迁移结论就没有解释空间。
- 至少要先确认 single-replica 时间、route 结果和 makespan 是可复现的。
### Step 3: 用统一口径比较收益与代价

- 分布式推理项目必须同时看路由结果、负载均衡、makespan 和通信代价，不能只挑单项吞吐收益下结论。
- `assignments` 回答请求是否按预期分散到不同副本。
- `makespan_ms` 和 `throughput_req_per_ms` 回答逻辑收益是否存在。
- `imbalance_ratio` 和 `comm_cost_ms` 回答收益是不是靠额外通信或严重不均衡换来的。
### Step 4: 输出迁移结论

- 分布式推理最终不是输出“逻辑模拟能不能跑”，而是输出这条迁移路线在当前 workload 下是否值得继续保留、微调或进入真实多卡部署。
- 最终结论建议统一为 `accept / tune / reject`。
- 若进入 `tune`，下一轮优先回路由策略、切分度和副本配置，而不是直接上真多卡部署。
#### 图解：66-70-79 如何收束到 81 分布式推理逻辑验证

```text
66 inference baseline -> 70 serving scheduler -> 79 distributed parallel benchmark
                         |
                         v
            81 distributed inference validation + migration decision
```
项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| baseline | workload、single-replica time、route policy | 保证迁移比较合法 |
| candidate | TP/PP 逻辑度、副本数、通信成本 | 解释分布式收益来源 |
| 对比 | makespan、throughput、imbalance、comm cost | 判断是否值得迁移 |
| 决策 | accept / tune / reject | 输出逻辑验证结论 |

```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：请求成本估算、分布式模拟、迁移决策
# 目标：把分布式推理验证整理成 baseline -> candidate -> decision 闭环

def estimate_request_cost(request: Dict[str, int], config: Dict[str, float]) -> Dict[str, float]:
    """null"""
    # prompt_tokens = ???；generated_tokens = ???；prefill_ms = ???；decode_ms = ???；
    # communication_ms = ???；e2e_ms = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

def simulate_distributed_inference(
    requests: List[Dict[str, int]], num_replicas: int, config: Dict[str, float]
) -> Dict[str, object]:
    """null"""
    # replica_loads = ???；makespan_ms = ???；imbalance_ratio = ???；
    # total_comm_ms = ???；throughput = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_distributed_inference_run(
    baseline: Dict[str, float], candidate: Dict[str, float], max_imbalance_ratio: float, max_comm_ratio: float
) -> Dict[str, object]:
    """null"""
    # latency_ok = ???；imbalance_ok = ???；comm_ok = ???；decision = ???；
    # reason = ???；next_action = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_distributed_inference_project():
    requests = [
        {'prompt_tokens': 128, 'generate_tokens': 64},
        {'prompt_tokens': 64, 'generate_tokens': 32},
    ]
    config = {
        'prefill_ms_per_token': 0.01,
        'decode_ms_per_token': 0.02,
        'comm_ms_per_token': 0.03,
        'pipeline_penalty_ms': 0.25,
        'tp_degree': 2,
        'pp_stages': 1,
    }

    cost = estimate_request_cost(requests[0], config)
    assert round(cost['prefill_ms'], 2) == 1.28
    assert round(cost['decode_ms'], 2) == 1.28
    assert round(cost['comm_ms'], 2) == 0.06
    assert round(cost['total_ms'], 2) == 2.62

    summary = simulate_distributed_inference(requests, num_replicas=2, config=config)
    assert summary['routing_strategy'] == 'least_loaded'
    assert summary['assignments'] == [0, 1]
    assert round(summary['single_replica_time_ms'], 2) == 3.93
    assert round(summary['makespan_ms'], 2) == 2.62
    assert round(summary['throughput_req_per_ms'], 4) == 0.7634
    assert round(summary['imbalance_ratio'], 4) == 0.6667
    assert round(summary['comm_cost_ms'], 2) == 0.09

    decision = recommend_distributed_inference_run(
        {'single_replica_time_ms': 3.93},
        summary,
        max_imbalance_ratio=0.8,
        max_comm_ratio=0.05,
    )
    assert decision['decision'] == 'accept'
    assert decision['next_action'] == 'promote_to_real_cluster_check'

    weak_candidate = {
        'single_replica_time_ms': 3.93,
        'makespan_ms': 3.10,
        'imbalance_ratio': 0.9,
        'comm_cost_ms': 0.09,
    }
    weak_decision = recommend_distributed_inference_run(
        {'single_replica_time_ms': 3.93},
        weak_candidate,
        max_imbalance_ratio=0.8,
        max_comm_ratio=0.05,
    )
    assert weak_decision['decision'] == 'tune'

    bad_candidate = {
        'single_replica_time_ms': 3.93,
        'makespan_ms': 4.30,
        'imbalance_ratio': 1.2,
        'comm_cost_ms': 0.40,
    }
    bad_decision = recommend_distributed_inference_run(
        {'single_replica_time_ms': 3.93},
        bad_candidate,
        max_imbalance_ratio=0.8,
        max_comm_ratio=0.05,
    )
    assert bad_decision['decision'] == 'reject'


test_distributed_inference_project()
print('测试通过：分布式推理逻辑验证模板可以工作。')
```

🛑 **STOP HERE** 🛑

请先尝试自己完成代码并跑通测试。如果你在 Colab 中运行，并且暂时没有思路，再继续看下面的参考答案。
## 参考代码与解析

```python
from typing import Dict, List


def estimate_request_cost(request: Dict[str, int], config: Dict[str, float]) -> Dict[str, float]:
    prompt_tokens = request['prompt_tokens']
    generate_tokens = request['generate_tokens']
    prefill_ms = prompt_tokens * config['prefill_ms_per_token']
    decode_ms = generate_tokens * config['decode_ms_per_token']
    comm_ms = max(config['tp_degree'] - 1, 0) * config['comm_ms_per_token'] * (
        prompt_tokens / 128.0 + generate_tokens / 64.0
    )
    pipeline_ms = max(config['pp_stages'] - 1, 0) * config['pipeline_penalty_ms']
    total_ms = prefill_ms + decode_ms + comm_ms + pipeline_ms
    return {
        'prefill_ms': round(prefill_ms, 4),
        'decode_ms': round(decode_ms, 4),
        'comm_ms': round(comm_ms, 4),
        'total_ms': round(total_ms, 4),
    }


def simulate_distributed_inference(
    requests: List[Dict[str, int]], num_replicas: int, config: Dict[str, float]
) -> Dict[str, object]:
    replica_loads = [0.0 for _ in range(num_replicas)]
    assignments = []
    single_replica_time_ms = 0.0
    comm_cost_ms = 0.0

    for request in requests:
        cost = estimate_request_cost(request, config)
        single_replica_time_ms += cost['total_ms']
        comm_cost_ms += cost['comm_ms']
        chosen_replica = min(range(num_replicas), key=lambda idx: replica_loads[idx])
        assignments.append(chosen_replica)
        replica_loads[chosen_replica] += cost['total_ms']

    makespan_ms = max(replica_loads) if replica_loads else 0.0
    throughput_req_per_ms = len(requests) / makespan_ms if makespan_ms else 0.0
    min_load = min(replica_loads) if replica_loads else 0.0
    avg_load = sum(replica_loads) / len(replica_loads) if replica_loads else 0.0
    imbalance_ratio = (makespan_ms - min_load) / avg_load if avg_load else 0.0

    return {
        'routing_strategy': 'least_loaded',
        'assignments': assignments,
        'single_replica_time_ms': round(single_replica_time_ms, 4),
        'makespan_ms': round(makespan_ms, 4),
        'throughput_req_per_ms': round(throughput_req_per_ms, 4),
        'imbalance_ratio': round(imbalance_ratio, 4),
        'comm_cost_ms': round(comm_cost_ms, 4),
    }


def recommend_distributed_inference_run(
    baseline: Dict[str, float], candidate: Dict[str, float], max_imbalance_ratio: float, max_comm_ratio: float
) -> Dict[str, object]:
    baseline_time = baseline.get('single_replica_time_ms', 0.0)
    candidate_makespan = candidate.get('makespan_ms', 0.0)
    comm_ratio = candidate.get('comm_cost_ms', 0.0) / baseline_time if baseline_time else 0.0

    if (
        candidate_makespan < baseline_time
        and candidate.get('imbalance_ratio', 1.0) <= max_imbalance_ratio
        and comm_ratio <= max_comm_ratio
    ):
        return {
            'decision': 'accept',
            'reason': '逻辑时延下降，且负载与通信代价都在可控范围内',
            'next_action': 'promote_to_real_cluster_check',
        }
    if candidate_makespan < baseline_time:
        return {
            'decision': 'tune',
            'reason': '已有逻辑收益，但负载均衡或通信比例还不够稳',
            'next_action': 'refine_routing_or_parallel_degree',
        }
    return {
        'decision': 'reject',
        'reason': 'candidate 没有形成可信的分布式推理收益',
        'next_action': 'fallback_to_single_replica_audit',
    }
```

### 解析

这页现在按 `estimate -> simulate -> decide` 的最小分布式推理迁移闭环组织，不再只是做单卡逻辑模拟。

#### TODO 1

- 实现方式：先分别估算 prefill、decode、TP 通信和 PP 流水惩罚，再合成单请求总时间。
- 关键点：这里的通信成本和流水惩罚不是精确运行时，而是帮助判断迁移是否可能被系统代价吃掉的逻辑估算。
- 项目意义：先把请求成本拆开，后面才能解释分布式推理的收益到底来自并行加速还是只是估算假象。

#### TODO 2

- 实现方式：在 least-loaded 路由下模拟请求分配，汇总 single-replica 总时间、makespan、吞吐和不均衡。
- 关键点：`single_replica_time_ms` 是单机顺序跑完这批请求的基线，`makespan_ms` 才是迁移后的真正系统完成时间。
- 项目意义：这一步把分布式推理从“能不能切分”转成“负载、吞吐和通信能否一起成立”的迁移比较。

#### TODO 3

- 实现方式：先比较基线与 makespan 的加速，再按 imbalance ratio 和 comm ratio 输出 `accept / tune / reject`。
- 关键点：`tune` 主要对应已有迁移收益，但路由策略、并行切分度或副本配置还没有一起收稳。
- 项目意义：分布式推理项目最后要回答的是“值不值得迁移到真实集群”，而不是只看单次逻辑模拟有没有加速。

## 相关阅读

以下资料按“分布式推理架构 → 推理系统实现 → 多卡证据”排列，用于把本节的路由、makespan 和通信成本连接到真实部署。

- [Megatron-LM 论文：大规模 Transformer 并行训练与推理基础](https://arxiv.org/abs/1909.08053)
- [vLLM 官方仓库](https://github.com/vllm-project/vllm)
- [SGLang 官方仓库](https://github.com/sgl-project/sglang)
- [PyTorch Distributed 官方文档](https://pytorch.org/docs/stable/distributed.html)
- [79 分布式并行基准](./79_Distributed_Parallel_Benchmark.md)
- [80 MoE 专家并行基准](./80_MoE_Expert_Parallel_Benchmark.md)
- [74 Profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
