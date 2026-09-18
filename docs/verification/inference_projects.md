# 66–72 推理项目验证清单

本页是 `Part 02` 推理项目的执行清单。它补充 Notebook 中的实验说明，不替代各节的题目、答案和策略解释。

## 验证顺序

```text
静态检查 → 答案区回归 → 真实 backend smoke test → JSON schema 检查 → 扩大 workload 后再下结论
```

## 1. 静态检查与答案区回归

在仓库根目录执行：

```bash
python -m py_compile \
  tools/inference_result_schema.py \
  tools/inference_project_runtime.py \
  tools/benchmark_inference_backend.py

python tools/test_notebook_answers.py \
  --dir 02_PyTorch_Algorithms \
  --mode answer

python tools/check_docs_links.py \
  --skip-convert --skip-mirror-check --skip-build
```

如果只验证单节：

```bash
python tools/test_notebook_answers.py \
  02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb \
  --mode answer
```

## 2. 运行入口与项目分级

| 项目 | 默认级别 | Notebook 开关 | 结果文件 | 验证边界 |
|---|---|---|---|---|
| 66 | Practice-P2 | `RUN_REAL_BACKEND = True` | `66_vllm_real.json` | 完整验证模型解析、dtype、端口、vLLM 生命周期和 benchmark |
| 67 | Practice-P1；可选 P2 | `RUN_REAL_BACKEND = True` | `67_quantized_deployment.json` | 默认是部署 smoke test；量化模型和量化启动参数需要另行配置 |
| 68 | Practice-P1；P2 扩展 | `RUN_BACKEND_SMOKE = True` | `68_backend_smoke.json` | 只能验证 baseline endpoint；不能据此宣称 speculative decoding 已启用 |
| 69 | Practice-P1；可选 P2 | `RUN_REAL_BACKEND = True` | `69_prefix_cache.json` | 必须确认 prefix cache 开关、命中率和失效开销，而不只是服务启动成功 |
| 70 | Practice-P1；可选 P2 | `RUN_REAL_BACKEND = True` | `70_scheduler.json` | 需要并发 workload；单次 smoke test 不能代表调度器整体收益 |
| 72 | Practice-P1；可选 P2 | Step 5 部署开关 | `72_model_deployment.json` | 固定 workload，比较具体 model / revision / artifact / backend / hardware；记录加载状态、TTFT、TPOT、吞吐、P99、显存和质量 |

在真实 GPU 环境中，Notebook 会通过 `tools/inference_project_runtime.py`：

- 从 `auto`、`modelscope`、`huggingface` 或 `local` 解析模型；
- 自动选择可用 dtype；
- 自动选择空闲端口并等待服务就绪；
- 运行 OpenAI-compatible benchmark；
- 把结果保存到 `benchmarks/results/`；
- 在 `finally` 中停止 backend。

66 使用兼容旧版 vLLM / 多环境 Notebook 的专用入口，但仍然复用相同的 benchmark 参数和 `normalized_result` 输出；67–70 使用共享 runtime helper。72 当前先完成配置审计和对照记录，真实 backend 启动与结果落盘作为后续增量。已接入 runtime 的入口都必须满足“启动失败不生成成功结论、服务结束后执行清理、结果写入仓库根目录”的要求。

从 Colab / ModelScope 打开 Notebook 时，先确保仓库已经 clone，并从仓库根目录运行；没有 GPU 时保持真实 backend 开关关闭。

## 3. 66 真实 backend 验证

在 66 节配置单元中设置：

```python
RUN_REAL_BACKEND = True
```

运行 Step 6。完成后检查：

```bash
ls -lh benchmarks/results/66_vllm_real.json
jq '.normalized_result' benchmarks/results/66_vllm_real.json
```

应能看到模型、backend、dtype、并发、TTFT、TPOT、E2E latency、吞吐和 decision 等字段。

## 4. 67–72 验证

分别打开对应 Notebook 的可选运行单元：

- 67：设置 `RUN_REAL_BACKEND = True`。若要验证真正量化部署，把 `MODEL_ID` 换成量化模型或本地量化目录；当前默认模型只证明服务链路。
- 68：设置 `RUN_BACKEND_SMOKE = True`。该结果是 speculative baseline 的 backend smoke test，真正 speculative 实验还需要 draft model 和 verify 能力。
- 69：设置 `RUN_REAL_BACKEND = True`，并确认 backend 的 prefix-cache 配置确实打开。
- 70：设置 `RUN_REAL_BACKEND = True`，至少使用并发 4 的 workload，再比较 TTFT、TPOT、吞吐和公平性。
- 72：在 Step 5 中打开部署开关，先固定 66 的 workload，再一次只改变模型、模型产物或 backend；必须记录 model ID、revision、格式、dtype、硬件和加载状态。

各节的自动化边界如下：

| 项目 | 自动完成 | 仍需学习者确认 |
|---|---|---|
| 66 | 模型解析、端口、服务生命周期、benchmark、结果保存 | vLLM 与当前 CUDA/驱动是否匹配 |
| 67 | 模型解析、服务 smoke、结果保存 | 量化格式、量化启动参数和质量回归 |
| 68 | baseline backend smoke、结果保存、服务清理 | draft model、接受率、verify 成本和真实 speculative 配置 |
| 69 | backend 启动、benchmark、结果保存、服务清理 | prefix cache 是否真正开启、命中率和失效开销 |
| 70 | backend 启动、并发 benchmark、结果保存、服务清理 | workload 规模、公平性、排队和长时间稳定性 |
| 72 | 模型与 backend 配置检查、对照记录和结果字段准备 | 具体模型是否支持目标 backend、artifact / kernel 是否匹配，以及真实质量和性能证据 |

因此，`RUN_BACKEND_SMOKE = True` 只适用于 68 的链路检查；它不会自动把 baseline smoke 升级成 speculative decoding 实验。67 的默认 Qwen 模型也只验证部署链路，不能直接代表量化收益。72 的 CPU 路径只验证配置和决策逻辑，不能替代真实模型与 backend 的部署证据。

检查结果文件：

```bash
ls -lh \
  benchmarks/results/67_quantized_deployment.json \
  benchmarks/results/68_backend_smoke.json \
  benchmarks/results/69_prefix_cache.json \
  benchmarks/results/70_scheduler.json \
  benchmarks/results/72_model_deployment.json
```

## 5. 统一 JSON schema 检查

66–72 的结果允许保留策略特有字段，但公共结果位于 `normalized_result`，版本为
`inference-benchmark/v1`。执行：

历史 66 结果可能只有旧版顶层字段。不要覆盖原始测量文件，可使用下面的工具另存一份规范化结果：

```bash
python tools/normalize_inference_results.py \
  benchmarks/results/66_vllm_real.json \
  --output benchmarks/results/66_vllm_real_normalized.json \
  --project 66 \
  --strategy vllm-real
```

新运行由 `tools/benchmark_inference_backend.py` 自动写入 `normalized_result`；迁移工具主要用于保留历史实验的可比字段。

```bash
for f in \
  benchmarks/results/66_vllm_real.json \
  benchmarks/results/67_quantized_deployment.json \
  benchmarks/results/68_backend_smoke.json \
  benchmarks/results/69_prefix_cache.json \
  benchmarks/results/70_scheduler.json \
  benchmarks/results/72_model_deployment.json; do
  echo "=== $f"
  jq -e '
    .normalized_result.schema_version and
    .normalized_result.config.model and
    .normalized_result.config.backend and
    .normalized_result.config.dtype and
    .normalized_result.config.batch and
    .normalized_result.config.concurrency and
    .normalized_result.config.cache_policy and
    .normalized_result.metrics.ttft_ms and
    .normalized_result.metrics.e2e_latency_ms and
    .normalized_result.decision
  ' "$f" >/dev/null && echo "schema OK" || echo "schema FAILED"
done
```

公共字段包括：

```text
model / backend / dtype / prompt_tokens / generated_tokens
batch / concurrency / cache_policy
TTFT / TPOT / e2e_latency / throughput / P99 / peak_memory
quality / decision
```

`68` 的 acceptance rate、`69` 的 cache hit rate、`70` 的公平性和队列开销放在
`strategy_metrics`，不修改公共 schema。

## 6. 结果解释边界

- smoke test 只能证明链路可运行，不能代表稳定线上性能；
- 真实 backend 结果必须同时保留模型、版本、dtype、workload 和硬件环境；
- 交互式服务优先看 TTFT / P99，批处理优先看吞吐；
- 只有在质量、资源预算和目标指标同时满足时，才允许输出 `accept`；
- 指标不稳定或证据不足时输出 `tune`，不要为了得到结论而扩大单次测试的解释范围。
