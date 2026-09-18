# 01. 为什么需要算子优化

## 页面目标

本页建立算子优化的共同问题：同一个数学表达式，为什么在不同的数据布局、访存路径和 kernel 实现下，会产生不同的时间、显存和端到端结果？

算子优化的对象不是源代码本身，而是从数学语义到硬件执行的映射。先确认输出正确，再判断局部 kernel 的改进能否传递到模型和服务层。

## 从数学语义到执行证据

| 层次 | 要回答的问题 | 需要留下的证据 |
|:---|:---|:---|
| 语义 | 输入、输出、shape、dtype 和边界条件是什么？ | 参考实现、测试用例、误差阈值 |
| kernel | 数据如何 load、计算、store？中间结果在哪里驻留？ | kernel 代码、编译配置、内存路径 |
| 硬件 | 哪些线程、warp、block 或 Tensor Core 执行它？ | GPU、指令路径、occupancy 或 trace |
| 系统 | 局部变快是否减少了端到端成本？ | 固定 workload、延迟、吞吐和显存 |

## 常见算子类型

不同算子暴露的瓶颈不同，不能用同一套 Kernel 结构解释所有问题。

| 类型 | 典型例子 | 主要优化对象 | 后续入口 |
|:---|:---|:---|:---|
| Elementwise | 激活、缩放、逐元素变换 | 合并访存、向量化、融合 | Task1–3 |
| Reduction | RMSNorm、Softmax、统计量 | 分块归约、同步、累加精度 | Task2–4 |
| GEMM | Linear、投影矩阵 | tile、Tensor Core、数据复用 | Task1–4 |
| Attention | QK、Softmax、PV | 分块、在线 Softmax、中间状态 | Task3–4 |
| Fusion | Norm + 激活、多个连续阶段 | 中间张量写回和 launch | Task3 |
| 不规则访问 | Gather、Scatter、稀疏路由 | layout、索引、负载均衡 | 扩展阅读 |

## 学习重点

先把 PyTorch 参考实现作为语义基线，再逐步替换为 Triton 或 CUDA kernel。每次只改变一个关键因素，并同时记录 shape、dtype、布局、warmup 和迭代次数。这样才能区分“结果正确”“kernel 变快”和“系统真的获益”。

## 与其他专题的连接

模型架构解释 Attention、MLP 和 MoE 的结构；本专题解释这些结构如何落成 kernel。编译与图优化解释图变换、IR 和 lowering；本专题先聚焦具体算子的 load/store、tile、fusion 和执行配置。推理、显存和性能分析专题提供端到端 workload、资源预算和证据解释。

## 本页出口

完成本页后，学习者应能写出一个算子的输入输出契约，指出参考实现与候选 kernel 的比较指标，并说明为什么 CPU 正确性测试不能替代 GPU 性能验证。
