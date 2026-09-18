"""Shared runtime logic for the memory budget decision project."""

import math
from typing import Dict, List


def validate_memory_budget(
    budget: Dict[str, float], quality_floor: Dict[str, float]
) -> Dict[str, object]:
    required_budget_keys = [
        "memory_cap_mb",
        "min_samples_per_s",
        "min_memory_saving_mb",
        "min_throughput_ratio",
    ]
    required_quality_keys = ["max_val_loss"]
    missing_keys = [key for key in required_budget_keys if key not in budget]
    missing_keys += [key for key in required_quality_keys if key not in quality_floor]
    numeric_values = {key: budget.get(key) for key in required_budget_keys}
    numeric_values.update({key: quality_floor.get(key) for key in required_quality_keys})
    invalid_keys = [
        key
        for key, value in numeric_values.items()
        if key not in missing_keys
        and (not isinstance(value, (int, float)) or not math.isfinite(value))
    ]
    memory_cap = budget.get("memory_cap_mb")
    min_throughput = budget.get("min_samples_per_s")
    min_saving = budget.get("min_memory_saving_mb")
    throughput_ratio = budget.get("min_throughput_ratio")
    max_loss = quality_floor.get("max_val_loss")
    if isinstance(memory_cap, (int, float)) and memory_cap <= 0:
        invalid_keys.append("memory_cap_mb")
    if isinstance(min_throughput, (int, float)) and min_throughput < 0:
        invalid_keys.append("min_samples_per_s")
    if isinstance(min_saving, (int, float)) and min_saving < 0:
        invalid_keys.append("min_memory_saving_mb")
    if isinstance(throughput_ratio, (int, float)) and not 0 <= throughput_ratio <= 1:
        invalid_keys.append("min_throughput_ratio")
    if isinstance(max_loss, (int, float)) and max_loss < 0:
        invalid_keys.append("max_val_loss")
    invalid_keys = list(dict.fromkeys(invalid_keys))
    return {
        "is_valid": not missing_keys and not invalid_keys,
        "missing_keys": missing_keys,
        "invalid_keys": invalid_keys,
    }


def summarize_memory_strategies(
    candidates: List[Dict[str, object]],
    budget: Dict[str, float],
    quality_floor: Dict[str, float],
) -> Dict[str, object]:
    feasible: List[Dict[str, object]] = []
    quality_failed = 0
    invalid_count = 0
    oom_count = 0
    evaluations = []
    seen_names = set()

    for candidate in candidates:
        name = candidate.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            invalid_count += 1
            evaluations.append(
                {
                    "name": name,
                    "status": "invalid",
                    "reasons": ["missing_or_duplicate_name"],
                }
            )
            continue
        seen_names.add(name)
        if candidate.get("status", "ok") == "oom":
            oom_count += 1
            evaluations.append({"name": candidate.get("name"), "status": "oom", "reasons": ["oom"]})
            continue
        memory = candidate.get("peak_memory_mb")
        throughput = candidate.get("samples_per_s")
        eval_loss = candidate.get("eval_loss", candidate.get("val_loss"))
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in (memory, throughput, eval_loss)
        ):
            invalid_count += 1
            evaluations.append({"name": candidate.get("name"), "status": "invalid", "reasons": ["missing_or_non_numeric_metric"]})
            continue
        memory_ok = memory <= budget["memory_cap_mb"]
        speed_ok = throughput >= budget["min_samples_per_s"]
        quality_ok = eval_loss <= quality_floor["max_val_loss"]
        if not quality_ok:
            quality_failed += 1
        reasons = []
        if not memory_ok:
            reasons.append("memory_over_budget")
        if not speed_ok:
            reasons.append("throughput_below_floor")
        if not quality_ok:
            reasons.append("quality_over_floor")
        evaluations.append({
            "name": candidate.get("name"),
            "status": "feasible" if not reasons else "rejected",
            "reasons": reasons,
        })
        if memory_ok and speed_ok and quality_ok:
            feasible.append(candidate)

    feasible.sort(
        key=lambda item: (
            item["peak_memory_mb"],
            -item["samples_per_s"],
            item.get("eval_loss", item.get("val_loss")),
        )
    )
    baseline = next(
        (
            item
            for item in candidates
            if item.get("name") == "baseline"
            and item.get("status", "ok") == "ok"
            and all(
                isinstance(item.get(key), (int, float))
                and math.isfinite(item.get(key))
                for key in ("peak_memory_mb", "samples_per_s")
            )
        ),
        None,
    )
    best = feasible[0] if feasible else None
    return {
        "candidate_count": len(candidates),
        "measured_count": len(candidates) - oom_count - invalid_count,
        "oom_count": oom_count,
        "invalid_count": invalid_count,
        "feasible_count": len(feasible),
        "best_candidate": best["name"] if best else None,
        "quality_failed_count": quality_failed,
        "feasible_names": [item["name"] for item in feasible],
        "baseline_peak_memory_mb": baseline["peak_memory_mb"] if baseline else None,
        "throughput_ratio": (
            best["samples_per_s"] / baseline["samples_per_s"]
            if baseline and best
            else None
        ),
        "baseline_available": baseline is not None,
        "best_peak_memory_mb": best["peak_memory_mb"] if best else None,
        "memory_saving_mb": (
            baseline["peak_memory_mb"] - best["peak_memory_mb"]
            if baseline and best
            else 0.0
        ),
        "min_memory_saving_mb": budget["min_memory_saving_mb"],
        "min_throughput_ratio": budget["min_throughput_ratio"],
        "evaluations": evaluations,
    }


def decide_memory_budget_project(summary: Dict[str, object]) -> Dict[str, object]:
    feasible_count = summary["feasible_count"]
    best_candidate = summary["best_candidate"]
    quality_failed_count = summary["quality_failed_count"]

    if not summary.get("baseline_available", False):
        return {
            "decision": "reject",
            "reason": "baseline_missing_or_invalid",
            "next_action": "rerun_baseline_before_comparing_candidates",
        }
    if feasible_count == 0:
        return {
            "decision": "reject",
            "reason": "no_strategy_meets_budget_and_quality",
            "next_action": "tighten_batch_or_rework_memory_plan",
        }
    meaningful_memory_gain = summary.get("memory_saving_mb", 0.0) >= summary[
        "min_memory_saving_mb"
    ]
    acceptable_throughput = (
        summary.get("throughput_ratio") is not None
        and summary["throughput_ratio"] >= summary["min_throughput_ratio"]
    )
    if (
        best_candidate != "baseline"
        and meaningful_memory_gain
        and acceptable_throughput
    ):
        return {
            "decision": "accept",
            "reason": "compression_strategy_is_best_feasible_option",
            "next_action": "promote_to_training_run",
        }
    if quality_failed_count > 0:
        return {
            "decision": "tune",
            "reason": "compression_needs_quality_recovery",
            "next_action": "adjust_checkpoint_scope_or_batch_plan",
        }
    if not meaningful_memory_gain:
        return {
            "decision": "tune",
            "reason": "memory_saving_below_meaningful_threshold",
            "next_action": "test_larger_pressure_or_optimizer_compression",
        }
    if not acceptable_throughput:
        return {
            "decision": "tune",
            "reason": "throughput_loss_exceeds_budget",
            "next_action": "reduce_compression_scope",
        }
    return {
        "decision": "tune",
        "reason": "baseline_still_best_under_current_budget",
        "next_action": "revisit_memory_strategy_mix",
    }
