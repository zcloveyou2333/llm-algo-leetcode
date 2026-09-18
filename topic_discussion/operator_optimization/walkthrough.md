# 算子优化问题链：从一个 Kernel 到端到端收益

这条问题链假设你接手了一个可以运行、但性能不稳定的算子。目标不是直接寻找“最快写法”，而是逐步确认：语义是否正确、瓶颈是否明确、局部收益能否传到模型和 workload。

![算子优化：从语义到端到端验证](../../docs/public/topic_discussion/operator_optimization/operator_optimization_overview.svg)

## 第一段：先写出参考实现和契约

从一个简单算子或模型组件开始，明确输入 shape、dtype、stride、边界和输出误差。Part 03 的[向量加法](../../03_Triton_Kernels/01_Triton_Vector_Addition.ipynb)和 [GEMM](../../03_Triton_Kernels/04_Triton_GEMM_Tutorial.ipynb)适合作为入口；参考实现先用 PyTorch 或 CPU 版本固定语义。

此时只回答一个问题：候选 kernel 在什么条件下算“正确”？不要先用 GPU 时间替代正确性测试。

## 第二段：把数据搬运路径画出来

当语义契约稳定后，观察 load、计算和 store 如何映射到 global memory、L2、shared memory、register 或 Tensor Core。Part 01 的 [Warp / Block / Shared Memory](../../01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics.ipynb) 和 Part 03 的 [Triton 内存模型](../../03_Triton_Kernels/12_Triton_Memory_Model_and_Debug.ipynb)用于解释布局与 tile。

这里的判断是：候选实现减少了哪类搬运，代价又转移到了哪里？

## 第三段：再决定是否融合

如果多个阶段反复写回中间张量，再比较 RMSNorm、Softmax 或 Attention 的融合方案：[融合 RMSNorm](../../03_Triton_Kernels/03_Triton_Fused_RMSNorm.ipynb)、[融合 Softmax](../../03_Triton_Kernels/06_Triton_Fused_Softmax.ipynb)和 [FlashAttention](../../03_Triton_Kernels/08_Triton_Flash_Attention.ipynb)可以作为不同复杂度的例子。

融合前先列出依赖和中间张量，融合后再检查 register、shared memory、occupancy 和边界处理。减少 kernel 数量不是充分条件。

## 第四段：把配置放回硬件执行模型

同一个 tile 或 fusion 在不同 GPU 上可能有不同结果。使用 Part 01 的 [Tensor Core](../../01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive.ipynb)，以及 Part 04 的 [CUDA / Triton / PyTorch 对照](../../04_CUDA_and_System_Optimization/21_CUDA_vs_Triton_vs_PyTorch.ipynb)和 [CUDA Stream](../../04_CUDA_and_System_Optimization/17_PyTorch_CUDA_Streams_and_Transfer.ipynb)，把 dtype、shape、同步和异步搬运纳入解释。

此时才开始做 GPU microbenchmark，并固定 warmup、重复次数、GPU 和编译目标。

## 第五段：搜索配置并解释 trace

在多个 shape 或多个 GPU 上，手工选一个配置通常不够。进入 [Triton Autotune 与 Profiling](../../03_Triton_Kernels/05_Triton_Autotune_and_Profiling.ipynb)，比较候选 tile、warp 数和 stages，并记录编译时间、缓存命中与运行时间。

profiling 的作用是验证瓶颈假设：如果 kernel 变快但端到端不变，应该检查 launch、同步和其他阶段，而不是继续盲目调 tile。

## 第六段：回到 Block 和固定 workload

最后用 [Triton Llama3 Block 项目](../../03_Triton_Kernels/13_Triton_Llama3_Block_Project.ipynb)把算子放回模型组件，再用固定 workload 比较端到端延迟、吞吐、显存、编译成本和质量。

项目报告至少保留：

- 参考实现与候选 kernel 的语义和数值对照；
- shape、dtype、GPU、编译目标、warmup 和重复次数；
- kernel microbenchmark 与端到端 benchmark；
- profiling 观察到的瓶颈和失败条件；
- `accept / tune / reject` 及其证据等级。

局部 kernel 的收益只有在目标 workload 中重复出现，才可以成为端到端优化结论。
