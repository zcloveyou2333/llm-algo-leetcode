# 07 Visual Assets

## 页面目标

这页负责收口 profiling 专题的图册资产。入口主图负责展示路线，正文图负责解释一个局部判断，不能用一张总图替代所有正文机制。

## 图册顺序

1. [`profiling_evidence_chain.svg`](../../docs/public/topic_discussion/profiling/profiling_evidence_chain.svg)
- 从问题提出到采证、归因、验证、行动的证据链总图

2. [`time_breakdown_trace.svg`](../../docs/public/topic_discussion/profiling/time_breakdown_trace.svg)
- operator / kernel / wait / launch 的时间拆分图

3. [`memory_timeline_diagnosis.svg`](../../docs/public/topic_discussion/profiling/memory_timeline_diagnosis.svg)
- memory timeline 和 residency 的诊断图

4. [`communication_overlap_map.svg`](../../docs/public/topic_discussion/profiling/communication_overlap_map.svg)
- 多卡等待与 overlap 关系图

5. [`benchmark_validation_board.svg`](../../docs/public/topic_discussion/profiling/benchmark_validation_board.svg)
- before / after、波动和回归设计图

6. [`action_decision_board.svg`](../../docs/public/topic_discussion/profiling/action_decision_board.svg)
- keep observing / inspect / optimize / revert 的决策图

## 当前状态

当前图册已覆盖 `01-06` 的主要入口。正文页先通过本页统一管理资产；后续替换图片时，只需要更新对应 SVG 和本页职责说明，不在多个正文页重复维护路径。

专题入口主图：[`profiling_evidence_overview.svg`](../../docs/public/topic_discussion/profiling/profiling_evidence_overview.svg)；路线图：[`profiling_task_route.svg`](../../docs/public/topic_discussion/profiling/profiling_task_route.svg)。路线图当前已补上 Task0“问题定义”节点，并与入口的 Task0–6 顺序一致。
