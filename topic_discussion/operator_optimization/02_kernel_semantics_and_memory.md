# 02. Kernel 语义与访存

## 页面目标

本页把“kernel 能运行”拆成两个问题：它是否保持了参考实现的语义，以及它是否用更合适的方式搬运数据。边界、mask、dtype、layout 和 tile 是同一条执行链上的不同观察点。

![Kernel 数据路径：搬运、复用与归约](../../docs/public/topic_discussion/operator_optimization/operator_memory_flow.svg)

## Kernel 契约

| 契约 | 需要明确 | 常见风险 |
|:---|:---|:---|
| Shape | 输入维度、广播规则、输出形状 | 边界 tile 读写越界 |
| Dtype | 输入、累加和输出 dtype | 精度变化或隐式转换 |
| Mask | 有效元素与 padding 的范围 | 无效元素参与计算 |
| Layout | 连续性、stride、tile 映射 | 非连续访问和错误索引 |
| 数值 | 误差指标和容忍阈值 | 只检查 shape，不检查数值 |

Reduction 和不规则访问还需要额外检查并发语义。多个线程共同写入同一结果时，必须说明归约顺序、同步点和是否使用 atomic；否则即使小规模测试通过，也可能出现竞态或不可复现结果。

| 机制 | 需要说明 | 常见失败 |
|:---|:---|:---|
| 并行归约 | 局部结果如何合并，累加 dtype 是什么 | 结果依赖线程顺序 |
| 同步 | 哪些数据在 barrier 前后可见 | 读到未完成的中间值 |
| Atomic | 多线程写同一地址是否安全 | 竞态、吞吐下降 |
| Mask 边界 | 无效 lane 是否参与 load、compute、store | 越界或 padding 污染结果 |

## 访存如何影响 kernel

GPU kernel 通常在全局显存、L2、shared memory、register 和 Tensor Core 之间搬运数据。优化方向不是单纯减少指令，而是让数据在更接近计算的位置被复用，并减少不必要的中间写回。

| 观察对象 | 主要问题 | 可比较的指标 |
|:---|:---|:---|
| 全局访存 | 是否合并、是否重复读取 | memory throughput、kernel time |
| tile | 单个 program 处理多大区域 | tile 数、边界浪费、工作集 |
| shared memory | 是否提高复用，是否引入同步 | bank conflict、occupancy |
| register | 中间值是否过多 | register pressure、spill |

## 学习顺序

先用 CPU 或 PyTorch 参考实现覆盖边界，再在 Triton 中实现 load/store、mask 和归约，最后用 GPU 观察布局与访存代价。Part 01 的 Warp、Block、Shared Memory 和 SRAM 内容用于解释机制；Part 03 的 Triton Memory Model 用于调试实际索引。涉及 warp primitive、共享内存同步或 atomic 时，再回补 Part 04 的 CUDA 页面。

## 本页出口

你应能解释一个候选 kernel 的输入输出契约、边界 mask、主要数据搬运路径，以及为什么“输出 shape 一致”还不足以说明两个实现等价。
