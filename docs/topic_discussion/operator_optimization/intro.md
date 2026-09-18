# 算子优化（Operator Optimization）

> 专题类型：主学习路线　主服务目标：Kernel 性能与端到端收益

## 页面导语

本专题面向希望把一个数学算子实现成可测量、可调优 kernel 的学习者。你会从算子语义和输入输出开始，逐步理解 tile、访存、融合、硬件执行和 autotune，最后把局部 kernel 结果接回模型 Block 与端到端 workload。想按性能现象选方向时，可使用[算子优化判断手册](./casebook.md)；想沿一个 kernel 项目连续推进时，可阅读[算子优化问题链](./walkthrough.md)。

路线以 Part 03 的 Triton 实现为主线，Part 01 提供硬件与执行模型，Part 04 补充 CUDA 和系统机制。学习重点不是记住某个“最快配置”，而是建立一条可复查的判断链：语义正确 → 数值对齐 → kernel 可执行 → microbenchmark 有证据 → 端到端收益成立。

![算子优化：从数学语义到端到端收益](../../public/topic_discussion/operator_optimization/operator_optimization_overview.svg)

算子优化先保证语义和数值正确，再通过 kernel、访存和执行证据判断局部改进是否真正传递到端到端 workload。

## 如何开始

推荐先具备 Part 02 的 Tensor、Attention 和 Block 基础，再按 Task0–6 推进。第一次学习时先完成 Task0–2，建立“输入输出—访存—布局”的共同口径；再进入 Task3–5，理解融合、硬件执行和搜索；最后用 Task6 完成项目验证。

没有 GPU 时，可以先完成算子语义、边界处理、CPU 正确性和成本模型；GPU 阶段再验证真实 kernel 时间、编译成本、autotune、显存和端到端收益。CPU 结果是机制与正确性证据，不应写成 GPU 加速结论。

- Part01：解释硬件、并行层次、访存和性能约束；
- Part03：实现和调试 Triton kernel，是本路线的主要实践入口；
- Part04：补充 CUDA、异步执行、共享内存、Tensor Core 和系统级优化；
- 原有[编译与图优化](../compiler_graph_optimization/intro.md)：负责 graph rewrite、IR、lowering 和编译器决策，不与本专题合并。

如果只是想定位瓶颈，先看[性能分析](../profiling/intro.md)；如果问题首先表现为请求延迟或 KV Cache，再转到[推理优化](../inference_optimization/intro.md)。

### Part 01–04 的支撑关系

Part 01、Part 03 和 Part 04 不是三条并列路线，而是分别承担“解释机制、实现 kernel、下探执行”的职责；Part 02 作为模型组件桥接层，把 RMSNorm、SwiGLU、Attention 和 Block 放回 Transformer 结构。学习时先用 Part 01 建立判断语言，再以 Part 03 完成主要实践，遇到模型组件时回看 Part 02，遇到硬件执行或系统级问题时进入 Part 04。

| 来源 | 在算子优化路线中的职责 | 主要回答的问题 | 进入时机 |
|:---|:---|:---|:---|
| Part 01 | 硬件与执行机制基础 | 数据经过哪些存储层、线程层次和指令路径？ | Task0–4 作为机制前置或回补 |
| Part 02 | 模型组件桥接 | 一个算子在 Transformer Block 的哪一段，哪些组件适合融合或分块？ | Task0、Task1、Task3、Task6 按需回补 |
| Part 03 | Triton 主实践线 | 如何把语义写成 kernel、进行融合、调试和 autotune？ | Task1–6 作为主要实现入口 |
| Part 04 | CUDA 与系统深化 | 如何控制更底层的执行、异步搬运、共享内存和系统代价？ | Task4–6 作为执行深化 |

因此，Part 01 和 Part 02 的理论、模拟或 CPU 结果不能直接替代 GPU 实测；Part 03 的 Triton kernel 是主实践证据，负责写出和调优 Kernel；Part 04 的 CUDA 实验负责解释 Kernel 如何在硬件上执行，并扩大适用范围。

## 主学习线与验证出口

`Task0–6` 是路线节点；表中的 Part 01 是机制支撑，Part 02 是模型组件桥接，Part 03 是主要实现入口，Part 04 是底层执行和系统扩展入口。下表按“要回答的问题 → 学习入口 → 验证出口”组织；专题正文负责解释机制和判断条件，不替代 Notebook 中的实现与 benchmark。

![算子优化学习路线：从语义正确到端到端验证](../../public/topic_discussion/operator_optimization/operator_optimization_task_route.svg)

上图先给出 Task0–6 的学习顺序；下表再提供每个任务的具体 Notebook、项目入口和正文索引。

| Task | 学习内容 | 主学习线 / 项目入口 | 学习顺序 | 专题正文 |
|:---|:---|:---|:---|:---|
| Task0 | 算子语义、GPU 执行与性能边界 | [Part 02 · 00 PyTorch Warmup](../../02_PyTorch_Algorithms/00_PyTorch_Warmup.md) → [Part 01 · 08 编程模型](../../01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.md) → [Part 01 · 15 CUDA 执行模型](../../01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.md) | 数学语义 → 并行执行 → 性能指标 | [01 为什么需要算子优化](./01_why_operator_optimization_matters.md) |
| Task1 | Triton Kernel 基础 | [Part 02 · 01 RMSNorm](../../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.md) → [Part 01 · 18 Triton Block 模型](../../01_Hardware_Math_and_Systems/18_Triton_Block_Model.md) → [Part 03 · 01 Triton 向量加法](../../03_Triton_Kernels/01_Triton_Vector_Addition.md) → [Part 03 · 04 Triton GEMM](../../03_Triton_Kernels/04_Triton_GEMM_Tutorial.md) | tile → load/store → kernel → 正确性 | [02 Kernel 语义与内存访问](./02_kernel_semantics_and_memory.md) |
| Task2 | 访存、布局与片上存储 | [Part 01 · 16 Warp / Block / Shared Memory](../../01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics.md) → [Part 01 · 24 SRAM 优化](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md) → [Part 03 · 12 Triton 内存模型](../../03_Triton_Kernels/12_Triton_Memory_Model_and_Debug.md) | global memory → SRAM → layout → occupancy | [02 Kernel 语义与内存访问](./02_kernel_semantics_and_memory.md) |
| Task3 | 算子融合与模型组件 Kernel | [Part 02 · 02 SwiGLU](../../02_PyTorch_Algorithms/02_SwiGLU_Activation.md) → [Part 02 · 04 Attention](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md) → [Part 01 · 19 算子融合基础](../../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md) → [Part 03 · 03 融合 RMSNorm](../../03_Triton_Kernels/03_Triton_Fused_RMSNorm.md) → [Part 03 · 06 融合 Softmax](../../03_Triton_Kernels/06_Triton_Fused_Softmax.md)；扩展 [Part 02 · 20 FlashAttention 模拟](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)、[Part 03 · 08 FlashAttention](../../03_Triton_Kernels/08_Triton_Flash_Attention.md) | 组件结构 → 依赖 → 中间张量 → fusion → 端到端 | [03 融合与 Kernel 组合](./03_fusion_and_kernel_composition.md) |
| Task4 | Tensor Core、CUDA 与异步执行 | [Part 01 · 23 Tensor Core 深入](../../01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive.md) → [Part 04 · 03 Tensor Core MMA](../../04_CUDA_and_System_Optimization/03_Tensor_Core_MMA_Programming.md) → [Part 04 · 04 Warp Level Primitives](../../04_CUDA_and_System_Optimization/04_Warp_Level_Primitives.md) → [Part 04 · 16 CUDA Shared Memory](../../04_CUDA_and_System_Optimization/16_CUDA_Shared_Memory_Optimization.md) | instruction → warp / block → shared memory → kernel | [04 CUDA 执行与硬件约束](./04_cuda_execution_and_hardware_constraints.md) |
| Task5 | Autotune、Profiling 与 Kernel 选择 | [Part 02 · 44 自动调优框架](../../02_PyTorch_Algorithms/44_Auto_Tuning_Framework.md) → [Part 03 · 05 Triton 自动调优](../../03_Triton_Kernels/05_Triton_Autotune_and_Profiling.md) → [Part 04 · 17 CUDA Stream](../../04_CUDA_and_System_Optimization/17_PyTorch_CUDA_Streams_and_Transfer.md) → [Part 04 · 18 CUDA Graph](../../04_CUDA_and_System_Optimization/18_CUDA_Graph_and_JIT_Compile.md)；扩展 [Part 01 · 13 性能分析与瓶颈定位](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md) | 候选参数 → 搜索 → trace → 异步执行 | [05 成本模型与性能分析](./05_cost_model_and_profiling.md) |
| Task6 | Block 与端到端项目验证 | [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md) → [Part 03 · 13 Triton Llama3 Block 项目](../../03_Triton_Kernels/13_Triton_Llama3_Block_Project.md) → [Part 04 · 21 CUDA / Triton / PyTorch 对照](../../04_CUDA_and_System_Optimization/21_CUDA_vs_Triton_vs_PyTorch.md) → [Part 02 · 74 Profiling 驱动的端到端优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md) | 算子 → Block → 实现路径 → workload → 决策 | [06 基准测试与项目验证](./06_benchmark_and_project_validation.md) |

![算子优化知识地图：语义、数据路径、硬件执行与验证证据](../../public/topic_discussion/operator_optimization/operator_optimization_knowledge_map.svg)

## 如何使用专题正文

正文页不是 Task 的逐页复述，而是把多个 Notebook 中的机制串成一条判断链。第一次学习时按 01→06 阅读；遇到具体问题时，直接进入对应正文页，再回到 Task 表中的 Notebook 完成验证。

| 正文职责 | 主要回答的问题 | 对应路线位置 |
|:---|:---|:---|
| 01–02 建立共同口径 | 算子要保持什么语义，数据如何经过 kernel 和存储层？ | Task0–2 |
| 03–04 解释优化机制 | 为什么 fusion、Tensor Core、shared memory 或 stream 可能有效？ | Task3–4 |
| 05 解释证据与成本 | 如何选择配置，并把 trace 解释成瓶颈证据？ | Task5 |
| 06 收束项目结论 | 如何在固定 workload 下比较、复现并决定 accept / tune / reject？ | Task6 |

## 核心与扩展分级

核心路径以 Part 03 的 Triton 实现为主，先验证算子语义和数值正确性，再比较固定 shape 下的 kernel；扩展路径引入 Part 01 的硬件解释和 Part 04 的 CUDA、动态 shape、Tensor Core、异步执行与端到端 workload。节点变少、单个 kernel 变快或编译成功，都不能单独证明系统收益。

| Task | 核心路径 | 扩展路径 | 学习顺序 | 运行环境 |
|:---|:---|:---|:---|:---|
| Task0 | Part 01 · 08、15 与指标定义 | 不同 GPU 执行模型对照 | 语义 → 执行 | CPU · PyTorch |
| Task1 | Triton 01、04 与 Part 01 · 18 | CUDA kernel、Tensor Core | tile → kernel | CPU 语义验证；可选单 GPU · Triton |
| Task2 | Part 01 · 16、24 与 Triton 12 | layout、occupancy、异步搬运 | memory → layout | CPU 逻辑验证；单 GPU · Triton |
| Task3 | Part 01 · 19、Triton 03/06 | FlashAttention、复杂 fusion | 依赖 → fusion | 单 GPU · Triton / CUDA |
| Task4 | Part 01 · 23、Part 04 · 03/04/16 | Tensor Core、warp、shared memory | kernel → execution | 单 GPU · CUDA |
| Task5 | Triton autotune 与 Part 04 · 17/18 | 多 shape、Stream、CUDA Graph、Profiling | 搜索 → 异步执行 → 归因 | 单 GPU · Triton / CUDA |
| Task6 | Triton Block 项目与 Part 04 · 21 对照 | Triton / CUDA / PyTorch 实现路径与端到端比较 | 局部 → 系统 → 路径选择 | 单 GPU · Triton / CUDA / Profiler |

## 按需回补：Part 01 共享前置

这些 Part 01 小节在本专题中只承担共享基础角色；Part 02 的模型组件页面和 Part 03 / Part 04 的主线入口已经列在上面的 Task 表中。它们可以服务于推理、显存和编译路线，不能把其中的理论或模拟直接写成某个 GPU 的实测结论。

| P1 前置 | 本路线使用它回答什么 | 不能直接推出什么 |
|:---|:---|:---|
| [Part 01 · 08 编程模型](../../01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.md)、[Part 01 · 15 CUDA 执行模型](../../01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.md) | Triton、CUDA、PyTorch 和 GPU 执行边界 | 某个实现一定更快 |
| [Part 01 · 16 Warp / Shared Memory](../../01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics.md)、[Part 01 · 23 Tensor Core](../../01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive.md) | 并行层次、同步和矩阵指令 | 一个 tile 配置适合所有 GPU |
| [Part 01 · 19 算子融合](../../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)、[Part 01 · 24 SRAM 优化](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md) | 中间张量、数据驻留和读写代价 | fusion 一定降低端到端延迟 |
| [Part 01 · 13 性能分析与瓶颈定位](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md) | 建立 kernel 和系统热点证据 | 单次 trace 能代表所有 workload |

Part 03 和 Part 04 的主线入口已经列在上面的 Task 表中；不需要为了进入专题而先完成两个 Part 的全部内容。只有当当前 Task 的机制或实验出现知识缺口时，再回补对应页面。

## 跨专题入口

大模型架构负责 Attention、MoE、MLA 等结构本身；本专题在结构落到 kernel、fusion、layout 或执行计划时介入。编译与图优化负责图变换、IR、lowering 和 backend 选择；本专题负责具体 kernel 的语义、访存和执行证据。推理和显存优化负责服务指标与资源预算，Task6 再把局部 kernel 结果接回端到端验证。

如果问题首先表现为请求延迟或 KV Cache，转到[推理优化](../inference_optimization/intro.md)；如果重点是峰值显存、重算或卸载，转到[显存优化](../memory_performance_tuning/intro.md)；如果需要系统化定位热点，先看[性能分析](../profiling/intro.md)；如果问题涉及图变换、IR 或 lowering，转到[编译与图优化](../compiler_graph_optimization/intro.md)。

## 项目产出

先完成 CPU-first 的参考实现、数值对齐和边界测试，再进行 GPU microbenchmark；确认 kernel 有收益后，再用固定 workload 检查端到端延迟、吞吐、显存和编译成本。每次实验至少记录输入 shape、dtype、基线实现、候选实现、GPU、后端版本、warmup、迭代次数和证据等级。

最小项目产物包括：正确性对照、kernel 测试表、配置与编译信息、GPU microbenchmark、端到端对照、失败条件和最终 `accept / tune / reject` 结论。

![算子优化证据流：从候选实现到项目决策](../../public/topic_discussion/operator_optimization/operator_optimization_evidence_flow.svg)

## 环境与验证

按实验层级选择环境，不需要为了阅读 Part 01 或 CPU 机制验证提前安装完整 CUDA 工具链。

| 实验层级 | 环境组合 | 适合验证的内容 | 不能直接推出的结论 |
|:---|:---|:---|:---|
| CPU 机制 | [base.txt](../../requirements/base.txt) + [torch-cpu.txt](../../requirements/torch-cpu.txt) | 参考实现、shape、mask、数值误差、成本模型 | GPU 带宽、真实 kernel 时间和 Tensor Core 收益 |
| GPU Triton | [gpu.txt](../../requirements/gpu.txt) | Triton kernel、显存、CUDA 时间、固定 shape 对照 | 其他 GPU 或所有 workload 都更快 |
| CUDA 深化 | [gpu.txt](../../requirements/gpu.txt) + Part 04 CUDA 工具链 | Stream、异步搬运、Shared Memory、Tensor Core 路径 | 单个 microbenchmark 必然带来端到端收益 |
| Profiling | [profiling.txt](../../requirements/profiling.txt) + GPU 环境 | trace、kernel 热点、occupancy、访存和等待 | 一次 trace 可以代表所有输入分布 |

Part 03 的 Triton 实践通常需要 GPU；Part 04 的 CUDA 页面还需要匹配的 CUDA 编译器和驱动。每次更换 GPU、PyTorch、CUDA、Triton 或编译选项，都应重新完成 smoke test，并在报告中记录版本。

CPU 可以验证数学语义、边界处理、输出对齐和部分成本模型；真实 Triton/CUDA kernel、autotune、Tensor Core、CUDA Graph 和性能结论需要 GPU。没有匹配 workload 或真实 trace 时，只能记录为优化假设，不能写成通用结论。

图像资产与维护规则见[算子优化图册](./07_visual_assets.md)。
