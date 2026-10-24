"""Generate the requested 48-case comparison without loading model weights.

Only explicitly named tensor payloads are exact. Worker estimates are heuristic;
measured peaks stay null until the CUDA benchmark has actually completed.
"""

import argparse
import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_worker.memory import GIB, estimate, text_token_bound


def build_cases(fixture, spec):
    cfg = spec["configs"]["transformer/config.json"]["content"]
    layers, heads, head_dim = cfg["num_layers"], cfg["num_attention_heads"], cfg["attention_head_dim"]
    hidden = heads * head_dim
    cache_coefficient = layers * 2 * heads * head_dim * 2
    components = spec["components"]
    resident = components["transformer"]["bf16_parameter_bytes"] + components["vae"]["bf16_parameter_bytes"]
    all_weights = spec["bf16_parameter_bytes"]
    cases = []
    for resolution, cfg_scale, references, cache, mode in itertools.product(
        (1024, 2048), (1, 2), (1, 4, 10), (True, False), ("sequential", "parallel")
    ):
        batch = fixture["num_outputs"] if mode == "parallel" else 1
        counts = fixture["counts_by_reference_count"][str(references)]
        target_tokens = (resolution // 16) ** 2
        image_tokens = references * target_tokens
        positive = counts["positive_non_image_tokens"] + image_tokens
        negative = counts["negative_non_image_tokens"] + image_tokens if cfg_scale > 1 else 0
        cache_bytes = batch * (positive + negative) * cache_coefficient if cache else 0
        prefix_max = max(positive, negative)
        prefill_tokens = prefix_max + target_tokens
        # In this pinned eager SwiGLU expression, activation(gate(x)), up(x),
        # and their out-of-place product coexist during multiplication. Do not
        # add QKV to this payload: attention temporaries need not still be live.
        intermediate = hidden * cfg["mlp_ratio"]
        cache_so_far = 0
        prefill_live_bounds = []
        for prefix in [positive] + ([negative] if negative else []):
            if cache:
                cache_so_far += batch * prefix * cache_coefficient
            prefill_live_bounds.append(cache_so_far + 3 * batch * (prefix + target_tokens) * intermediate * 2)
        later_qkv = batch * (3 * target_tokens + 2 * prefix_max) * hidden * 2 if cache else batch * prefill_tokens * hidden * 3 * 2
        later_mlp_triplet = 3 * batch * (target_tokens if cache else prefill_tokens) * intermediate * 2
        eager_lower_bound = max(all_weights, resident + max(*prefill_live_bounds, cache_bytes + max(later_qkv, later_mlp_triplet)))
        need = estimate(
            [(fixture["source_reference_width"], fixture["source_reference_height"])] * references,
            resolution, resolution, resolution, len(fixture["prompt"]), cfg_scale > 1,
            cache, fixture["num_outputs"],
            positive_text_tokens=text_token_bound(fixture["prompt"], references),
            negative_text_tokens=text_token_bound(fixture["negative_prompt"], references),
        )
        denoise_estimate = need.parallel_total_bytes if mode == "parallel" else need.total_bytes
        decode_estimate = need.parallel_decoding_bytes if mode == "parallel" else need.decoding_bytes
        projected_estimate = max(
            all_weights + need.conditioning_bytes,
            resident + denoise_estimate,
            resident + decode_estimate,
        )
        row = {
            "case_id": f"{resolution}-cfg{cfg_scale}-ref{references}-kv{'on' if cache else 'off'}-{mode}",
            "output_width": resolution, "output_height": resolution,
            "reference_processing_resolution": resolution, "reference_count": references,
            "cfg_scale": cfg_scale, "kv_cache": cache, "execution": mode,
            "outputs_per_job": fixture["num_outputs"], "physical_batch_size": batch,
            "num_inference_steps": fixture["num_inference_steps"],
            "positive_non_image_tokens": counts["positive_non_image_tokens"],
            "negative_non_image_tokens": counts["negative_non_image_tokens"] if cfg_scale > 1 else 0,
            "target_tokens": target_tokens, "reference_tokens": image_tokens,
            "positive_prefix_tokens": positive, "negative_prefix_tokens": negative,
            "exact_persistent_kv_bytes": cache_bytes,
            "exact_bf16_parameter_bytes_all_components": all_weights,
            "exact_bf16_parameter_bytes_dit_and_vae": resident,
            "parameter_and_kv_peak_lower_bound_bytes": max(all_weights, resident + cache_bytes),
            "identified_eager_tensors_peak_lower_bound_bytes": eager_lower_bound,
            "prefill_single_branch_qkv_payload_bytes": batch * prefill_tokens * hidden * 3 * 2,
            "later_step_single_branch_qkv_payload_bytes": batch * (target_tokens if cache else prefill_tokens) * hidden * 3 * 2,
            "largest_single_mlp_expansion_payload_bytes": batch * prefill_tokens * hidden * cfg["mlp_ratio"] * 2,
            "prefill_eager_swiglu_three_tensor_payload_bytes": 3 * batch * prefill_tokens * intermediate * 2,
            "later_step_eager_swiglu_three_tensor_payload_bytes": later_mlp_triplet,
            "later_attention_input_qkv_payload_bytes": later_qkv,
            "worker_bound_persistent_kv_bytes": need.parallel_kv_cache_bytes if mode == "parallel" else need.kv_cache_bytes,
            "worker_estimated_conditioning_request_bytes": need.conditioning_bytes,
            "worker_estimated_denoising_request_bytes": denoise_estimate,
            "worker_estimated_decoding_request_bytes": decode_estimate,
            "worker_heuristic_parameter_plus_request_peak_bytes": projected_estimate,
            "measurement_status": "not_measured",
            "measured_job_peak_allocated_bytes": None,
            "measured_job_peak_reserved_bytes": None,
            "sampled_process_peak_bytes": None,
        }
        cases.append(row)
    # This describes the production guard, which chooses a route automatically.
    # The benchmark instead forces the sequential/parallel route in each row.
    # Capacity is illustrative: no driver, buffer or other-process reservation.
    by_id = {case["case_id"]: case for case in cases}
    for case in cases:
        seq = by_id[case["case_id"].replace("-parallel", "-sequential")]
        par = by_id[case["case_id"].replace("-sequential", "-parallel")]
        capacity = 80 * GIB
        case["ideal_80gib_guard_route_for_four_outputs"] = (
            "reject" if seq["worker_heuristic_parameter_plus_request_peak_bytes"] > capacity
            else "parallel" if par["worker_heuristic_parameter_plus_request_peak_bytes"] <= capacity
            else "sequential"
        )
        case["forced_route_lower_bound_exceeds_80gib"] = (
            case["identified_eager_tensors_peak_lower_bound_bytes"] > capacity
        )
        case["ideal_80gib_final_route_for_four_outputs"] = case["ideal_80gib_guard_route_for_four_outputs"]
    return cases


def merge_measurements(cases, results_dir):
    fingerprint = None
    for case in cases:
        path = results_dir / (case["case_id"] + ".json")
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        if value.get("model_revision") != "790c92633540aa0cb11d9abf19eb46d861714758":
            raise ValueError(f"Benchmark model revision mismatch: {path}")
        for key in ("case_id", "output_width", "output_height", "reference_count", "reference_processing_resolution",
                    "cfg_scale", "kv_cache", "execution", "outputs_per_job", "num_inference_steps"):
            if value.get("case", {}).get(key) != case[key]:
                raise ValueError(f"Benchmark case mismatch for {path}: {key}")
        case["measurement_status"] = value["status"]
        runs = [r for r in value.get("runs", []) if r.get("status") == "ok" and r["label"].startswith("measured_")]
        if (value["status"] != "ok" or not value.get("completed") or not runs
                or len(runs) != value.get("expected_measured_repeats")):
            if value["status"] == "ok":
                case["measurement_status"] = "incomplete_measurements"
            continue
        if not all(r.get("fixture_cache_match") for r in runs):
            raise ValueError(f"Benchmark fixture mismatch: {path}")
        current = value.get("runtime_fingerprint")
        if not current:
            raise ValueError(f"Missing GPU/runtime fingerprint: {path}")
        if fingerprint is not None and current != fingerprint:
            raise ValueError(f"Cannot compare measurements from different GPUs/runtimes: {path}")
        fingerprint = current
        case["measurement_runtime_fingerprint"] = current
        case["measured_repeats"] = len(runs)
        for field in ("measured_job_peak_allocated_bytes", "measured_job_peak_reserved_bytes"):
            samples = [r[field] for r in runs]
            case[field] = max(samples)
            case[field + "_median"] = statistics.median(samples)
            case[field + "_min"] = min(samples)
        sampled = [r["sampled_process_peak_bytes"] for r in runs if r.get("sampled_process_peak_bytes") is not None]
        case["sampled_process_peak_bytes"] = max(sampled) if sampled else None
        case["measurement_status"] = "measured"
    return cases


def write_report(cases, fixture, destination):
    out = [
        "# Qwen-Image-2.1: 48-case VRAM comparison", "",
        "Generated by `python scripts/vram_matrix.py`. All sizes below are **GiB**,",
        "where 1 GiB = 1,073,741,824 bytes. CSV/JSON preserve integer byte values.", "",
        "Analytical generation leaves GPU measurements missing. Use `--results-dir` to join actual benchmark results.",
        "Exact tensor payloads, lower bounds, and worker heuristic projections are",
        "separate columns. See [the measurement protocol](VRAM_MEASUREMENT.md).", "",
        "## Fixed workload", "",
        "- Four outputs per job: four batch-one calls in sequential mode, or one batch-four call in parallel mode.",
        "- References are processed at the same resolution as output: 1024 or 2048 square.",
        "- Original pinned base checkpoint, BF16, 40 steps, seed 42 plus output index.",
        "- Default `text_encoder` offload policy; VAE tiling and slicing enabled.",
        "- CFG=2 uses the worker's default empty negative prompt; CFG=1 has no negative branch.",
        "- Actual tokenizer/template counts, not a fixed assumed text length.", "",
        "Prompt: " + fixture["prompt"], "",
        "| References | Positive non-image tokens | Negative non-image tokens when CFG=2 |",
        "|---:|---:|---:|",
    ]
    for n, counts in fixture["counts_by_reference_count"].items():
        out.append(f"| {n} | {counts['positive_non_image_tokens']} | {counts['negative_non_image_tokens']} |")
    out += [
        "", "## How to read the columns", "",
        "- **K/V exact:** logical persistent DiT prefix K/V for this workload. Zero with caching off means no persistent prefix cache, not zero VRAM.",
        "- **Weights + K/V floor:** `max(all BF16 parameters, resident DiT/VAE parameters + K/V)`. This is only a peak lower bound under default encoder offload, excluding activations, buffers, context, workspaces, and allocator overhead.",
        "- **Eager tensor floor:** a stronger lower bound including the known out-of-place SwiGLU intermediate tensors at last-layer prefill, or cached-step attention/MLP payloads. It accounts for which CFG caches are populated at that point. It applies to this pinned eager implementation, not a fused/compiled alternative. It still excludes many live tensors and all non-PyTorch allocations.",
        "- **Worker projection:** the current admission formulas plus parameter payload for each phase; the maximum is shown. It is an uncalibrated estimate, not a measurement or guaranteed upper bound.",
        "- **Later QKV:** payload for the largest single branch's Q/K/V projection outputs in one DiT layer on a later step. It is one activation component, not the full working set. CFG branches execute sequentially.",
        "- **Measured peak:** unavailable until a successful CUDA benchmark. Missing data is never filled with zero.", "",
        "All-component BF16 parameter payload: **30.212083526 GiB**.",
        "DiT + VAE BF16 parameter payload: **13.882043101 GiB**.",
        "Text/template changes alter the small exact K/V offsets shown here.",
        "", "**80 GiB guard route** models a wholly available 80 GiB device using parameter payloads only;",
        "driver/context memory, buffers, other processes and fragmentation reduce actual headroom.",
        "The VRAM guard column reports memory admission; the final route also applies the inline output budget.",
        "Both columns describe four outputs and are repeated across the two forced benchmark rows.",
        "Admission is not a successful GPU measurement. See [the failure/cost audit](RUNPOD_RISK_REVIEW.md).",
    ]
    for resolution in (1024, 2048):
        out += ["", f"## {resolution} × {resolution} output and reference processing", "",
                "| CFG | Inputs | KV | Forced execution | K/V exact | Eager tensor floor | Worker projection | 80 GiB VRAM guard | Final route | Measured allocated peak |",
                "|---:|---:|---|---|---:|---:|---:|---|---|---|"]
        for c in cases:
            if c["output_width"] != resolution:
                continue
            measured = (f"{c['measured_job_peak_allocated_bytes']/GIB:.6f}"
                        if c["measured_job_peak_allocated_bytes"] is not None else c["measurement_status"])
            out.append(
                f"| {c['cfg_scale']} | {c['reference_count']} | {'on' if c['kv_cache'] else 'off'} | {c['execution']} "
                f"| {c['exact_persistent_kv_bytes']/GIB:.6f} "
                f"| {c['identified_eager_tensors_peak_lower_bound_bytes']/GIB:.6f} "
                f"| {c['worker_heuristic_parameter_plus_request_peak_bytes']/GIB:.6f} "
                f"| {c['ideal_80gib_guard_route_for_four_outputs']} "
                f"| {c['ideal_80gib_final_route_for_four_outputs']} "
                f"| {measured} |"
            )
    out += [
        "", "## Consequences", "",
        "- Cache-on BF16 storage scales with reference count, reference area, output batch, and number of CFG branches.",
        "- CFG scale 2 activates a second branch; scale 2 does not mean twice the arithmetic of every pipeline stage or twice every transient allocation.",
        "- Sequential mode retains one output's DiT K/V at a time. Parallel mode stores four copies in this pinned implementation.",
        "- Cache-off repeats full-prefix DiT computation at every step. Its memory use cannot be obtained by merely subtracting cache bytes from a measured cache-on peak.",
        "- Rows with an identified-tensor lower bound exceeding a GPU's actual capacity cannot complete under this placement policy and eager implementation. A floor below capacity does not establish that they fit.",
        "- The worker's byte-based token bound is intentionally larger than this fixture's actual token counts. Both exact K/V and the current worker's bounded K/V are retained in CSV/JSON.", "",
        "- Eleven forced rows have lower bounds above 80 GiB. Under the idealized 80 GiB guard calculation, every one is rejected or routed to sequential execution before inference.",
        "- Phase memory admission routes the 24 distinct requests as 13 parallel, six sequential, and five rejected. Response size does not change generation routing. Each appears twice in this matrix.",
        "- Actual encoded responses are compressed, resized, or partially delivered after generation to fit the inline cap. When VRAM requires sequential generation, a partial delivery can stop remaining passes to avoid wasting compute.",
        "- These checks cover VRAM admission for the fixed fixture. Long positional layouts, malformed inputs, oversized encoded outputs and deployment failures need separate checks.", "",
        "Sources: pinned [DiT implementation](https://github.com/huggingface/diffusers/blob/e0118ade2f60234c41bacf40330a7e2f61108849/src/diffusers/models/transformers/transformer_qwenimage21.py),",
        "[pipeline](https://github.com/huggingface/diffusers/blob/e0118ade2f60234c41bacf40330a7e2f61108849/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py),",
        "[tensor inventory](qwen-image-2.1-spec.json), and [worker estimate implementation](../qwen_worker/memory.py).", "",
    ]
    destination.write_text("\n".join(out), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "vram")
    parser.add_argument("--results-dir", type=Path, help="Join completed benchmark JSON files; never impute missing peaks")
    args = parser.parse_args()
    fixture = json.loads((ROOT / "docs/vram/fixture.json").read_text())
    spec = json.loads((ROOT / "docs/qwen-image-2.1-spec.json").read_text())
    cases = build_cases(fixture, spec)
    if args.results_dir:
        cases = merge_measurements(cases, args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "measurement_status": "analytical_with_optional_measurements", "assumptions": fixture,
               "model_revision": spec["model_revision"], "diffusers_commit": spec["diffusers_commit"],
               "case_count": len(cases), "cases": cases}
    (args.output_dir / "cases.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output_dir / "cases.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(key for case in cases for key in case))
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)
    write_report(cases, fixture, ROOT / "docs/VRAM_COMPARISON.md")
    measured = sum(c['measurement_status'] == 'measured' for c in cases)
    print(f"Wrote {len(cases)} cases; {measured} measured. Missing peaks remain null.")


if __name__ == "__main__":
    main()
