import argparse
import csv
import json
import math
import os
import random
import re
import sys
from datetime import datetime
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import transformers

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import flatquant.data_utils as data_utils
import flatquant.flat_utils as flat_utils
import flatquant.model_utils as model_utils
import flatquant.quant_utils as quant_utils
import flatquant.utils as utils
from flatquant.flat_linear import FlatQuantizedLinear


LINEAR_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
REP_LAYERS = [0, 2, 3, 20, 29, 30, 34, 35]
REP_LINEAR_NAMES = ["q_proj", "o_proj", "up_proj", "down_proj"]


def make_flatquant_args(args, exp_name):
    return SimpleNamespace(
        model=args.model,
        hf_token=args.hf_token,
        seed=args.seed,
        w_bits=4,
        a_bits=4,
        q_bits=16,
        k_bits=4,
        v_bits=4,
        w_asym=False,
        a_asym=False,
        q_asym=False,
        k_asym=True,
        v_asym=True,
        w_groupsize=-1,
        a_groupsize=-1,
        q_groupsize=-1,
        k_groupsize=128,
        v_groupsize=128,
        gptq=False,
        gptq_mse=False,
        percdamp=0.01,
        act_order=False,
        epochs=15,
        cali_dataset=args.cali_dataset,
        nsamples=args.nsamples,
        cali_bsz=4,
        flat_lr=5e-3,
        cali_trans=True,
        add_diag=True,
        lwc=True,
        lac=True,
        resume=False,
        save_matrix=False,
        reload_matrix=False,
        matrix_path=None,
        diag_init="sq_style",
        diag_alpha=0.3,
        warmup=False,
        deactive_amp=True,
        direct_inv=True,
        separate_vtrans=False,
        output_dir=args.output_root,
        exp_name=exp_name,
        exp_dir=os.path.join(args.output_root, exp_name),
    )


def load_flat_parameters_from_path(model, path):
    flat_parameters = torch.load(os.path.join(path, "flat_parameters.pth"), map_location="cpu")
    layers = model.model.layers
    for i in range(len(flat_parameters.keys())):
        layers[i].load_state_dict(flat_parameters[i], strict=False)


def module_short_name(name):
    if ".self_attn." in name:
        return name.split(".self_attn.", 1)[1].split(".", 1)[0]
    if ".mlp." in name:
        return name.split(".mlp.", 1)[1].split(".", 1)[0]
    return name.split(".")[-1]


def layer_index_from_name(name):
    parts = name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers":
            return int(parts[idx + 1])
    return -1


def get_target_modules(model, state_name):
    targets = {}
    if state_name == "fp":
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name != "lm_head" and any(f".{x}" in name for x in LINEAR_NAMES):
                if layer_index_from_name(name) in REP_LAYERS and module_short_name(name) in REP_LINEAR_NAMES:
                    targets[name] = module
    else:
        for name, module in model.named_modules():
            if isinstance(module, FlatQuantizedLinear) and any(name.endswith(f".{x}") for x in LINEAR_NAMES):
                if layer_index_from_name(name) in REP_LAYERS and module_short_name(name) in REP_LINEAR_NAMES:
                    targets[name] = module
    return targets


def sample_abs_values(x, max_count):
    x = x.detach().float().abs().flatten()
    if x.numel() == 0:
        return np.array([], dtype=np.float32)
    if x.numel() > max_count:
        idx = torch.randperm(x.numel(), device=x.device)[:max_count]
        x = x[idx]
    return x.cpu().numpy()


def sample_token_scores(x, max_count):
    x = x.detach().float().reshape(-1, x.shape[-1]).abs()
    score = x.max(dim=1)[0] / x.mean(dim=1).clamp(min=1e-12)
    if score.numel() > max_count:
        idx = torch.randperm(score.numel(), device=score.device)[:max_count]
        score = score[idx]
    return score.cpu().numpy()


def collect_distribution_samples(args, state_name, flat_path, dataloader):
    fq_args = make_flatquant_args(args, state_name)
    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    if state_name != "fp":
        model = apply_flatquant_to_model(fq_args, model)
        load_flat_parameters_from_path(model, flat_path)
        flat_utils.reparameterize_model(model)
        quant_utils.set_quantizer_state(model, enable=False)
    model.to(utils.DEV)
    model.eval()

    targets = get_target_modules(model, state_name)
    samples = {}
    for name, module in targets.items():
        weight = module.weight if state_name == "fp" else module.linear.weight
        samples[(state_name, name, "weight_abs")] = sample_abs_values(weight, args.max_weight_samples)

    act_abs = {name: [] for name in targets}
    token_scores = {name: [] for name in targets}
    hooks = []
    for name, module in targets.items():
        def hook(mod, inp, out, module_name=name):
            x = inp[0] if isinstance(inp, tuple) else inp
            act_abs[module_name].append(sample_abs_values(x, args.max_activation_samples_per_batch))
            token_scores[module_name].append(sample_token_scores(x, args.max_token_samples_per_batch))
        hooks.append(module.register_forward_hook(hook))

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= args.nsamples:
                break
            model(batch[0].to(utils.DEV))

    for hook in hooks:
        hook.remove()

    for name in targets:
        samples[(state_name, name, "activation_abs")] = concat_and_limit(act_abs[name], args.max_activation_samples_total)
        samples[(state_name, name, "token_score")] = concat_and_limit(token_scores[name], args.max_token_samples_total)

    del model
    utils.cleanup_memory(verbose=False)
    torch.cuda.empty_cache()
    return samples


def concat_and_limit(arrays, max_count):
    arrays = [x for x in arrays if len(x) > 0]
    if not arrays:
        return np.array([], dtype=np.float32)
    values = np.concatenate(arrays).astype(np.float32, copy=False)
    if len(values) > max_count:
        idx = np.random.choice(len(values), size=max_count, replace=False)
        values = values[idx]
    return values


def hist_prob(values, bins):
    hist, _ = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64) + 1e-12
    return hist / hist.sum()


def kl_div(p, q):
    return float(np.sum(p * np.log(p / q)))


def js_div(p, q):
    m = 0.5 * (p + q)
    return float(0.5 * kl_div(p, m) + 0.5 * kl_div(q, m))


def compute_distribution_divergences(samples):
    rows = []
    pairs = [("fp", "baseline"), ("baseline", "svd_mix_linear"), ("fp", "svd_mix_linear")]
    keys = sorted({(name, kind) for (_, name, kind) in samples.keys()})
    for name, kind in keys:
        available = {state: samples.get((state, name, kind)) for state in ["fp", "baseline", "svd_mix_linear"]}
        pooled = [v for v in available.values() if v is not None and len(v) > 0]
        if len(pooled) < 2:
            continue
        pooled_values = np.concatenate(pooled)
        upper = float(np.quantile(pooled_values, 0.999))
        if upper <= 0:
            continue
        bins = np.linspace(0.0, upper, 129)
        for left, right in pairs:
            lv = available.get(left)
            rv = available.get(right)
            if lv is None or rv is None or len(lv) == 0 or len(rv) == 0:
                continue
            p = hist_prob(np.clip(lv, 0.0, upper), bins)
            q = hist_prob(np.clip(rv, 0.0, upper), bins)
            rows.append({
                "layer": layer_index_from_name(name),
                "linear": module_short_name(name),
                "name": name,
                "kind": kind,
                "left": left,
                "right": right,
                "kl_left_right": kl_div(p, q),
                "kl_right_left": kl_div(q, p),
                "js": js_div(p, q),
                "hist_upper_p999": upper,
            })
    return rows


def parse_training_log(path):
    pattern = re.compile(
        r"layer (\d+) lwc lac iter 14,.*optimized_loss: ([0-9.eE+-]+), "
        r"plain_mse: ([0-9.eE+-]+), head_proxy_mse: ([0-9.eE+-]+)"
    )
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                layer = int(match.group(1))
                rows[layer] = {
                    "optimized_loss": float(match.group(2)),
                    "plain_mse": float(match.group(3)),
                    "head_proxy_mse": float(match.group(4)),
                }
    return rows


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_alignment_rows(args):
    baseline_log = parse_training_log(args.baseline_log)
    svd_log = parse_training_log(args.svd_log)
    activation_rows = read_csv_rows(os.path.join(args.prev_diag_dir, "activation_stats_all.csv"))
    weight_rows = read_csv_rows(os.path.join(args.prev_diag_dir, "weight_stats_all.csv"))
    by_kind = {"activation": activation_rows, "weight": weight_rows}

    rows = []
    for layer in sorted(set(baseline_log.keys()) & set(svd_log.keys())):
        base = baseline_log[layer]
        svd = svd_log[layer]
        plain_delta = svd["plain_mse"] - base["plain_mse"]
        head_delta = svd["head_proxy_mse"] - base["head_proxy_mse"]
        plain_rel = plain_delta / max(base["plain_mse"], 1e-12)
        head_rel = head_delta / max(base["head_proxy_mse"], 1e-12)
        for linear in LINEAR_NAMES:
            row = {
                "layer": layer,
                "linear": linear,
                "plain_mse_delta": plain_delta,
                "plain_mse_rel_delta": plain_rel,
                "head_proxy_mse_delta": head_delta,
                "head_proxy_mse_rel_delta": head_rel,
                "head_proxy_improved": head_delta < 0,
                "plain_mse_improved": plain_delta < 0,
            }
            for kind, source_rows in by_kind.items():
                base_metric = next((r for r in source_rows if r["state"] == "baseline" and int(r["layer"]) == layer and r["linear"] == linear), None)
                svd_metric = next((r for r in source_rows if r["state"] == "svd_mix_linear" and int(r["layer"]) == layer and r["linear"] == linear), None)
                if not base_metric or not svd_metric:
                    continue
                if kind == "activation":
                    metric_pairs = [
                        ("token_score_p99", True),
                        ("channel_max_over_median", True),
                        ("a4_sym_token_mse", True),
                        ("a4_sym_token_sqnr_db", False),
                    ]
                else:
                    metric_pairs = [
                        ("abs_max_over_p99", True),
                        ("per_out_max_over_median", True),
                        ("w4_sym_per_out_mse", True),
                        ("w4_sym_per_out_sqnr_db", False),
                    ]
                for metric, lower_better in metric_pairs:
                    delta = float(svd_metric[metric]) - float(base_metric[metric])
                    row[f"{kind}_{metric}_delta"] = delta
                    row[f"{kind}_{metric}_improved"] = delta < 0 if lower_better else delta > 0
            rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_divergences(rows):
    summary = {}
    for kind in sorted({row["kind"] for row in rows}):
        summary[kind] = {}
        for pair in [("fp", "baseline"), ("baseline", "svd_mix_linear"), ("fp", "svd_mix_linear")]:
            vals = [row["js"] for row in rows if row["kind"] == kind and row["left"] == pair[0] and row["right"] == pair[1]]
            summary[kind][f"{pair[0]}__{pair[1]}"] = {
                "mean_js": float(np.mean(vals)) if vals else 0.0,
                "median_js": float(np.median(vals)) if vals else 0.0,
                "max_js": float(np.max(vals)) if vals else 0.0,
                "count": len(vals),
            }
    return summary


def summarize_alignment(rows):
    summary = {
        "module_count": len(rows),
        "head_proxy_improved_count": sum(bool(row["head_proxy_improved"]) for row in rows),
        "plain_mse_improved_count": sum(bool(row["plain_mse_improved"]) for row in rows),
    }
    for key in [
        "activation_token_score_p99_improved",
        "activation_a4_sym_token_sqnr_db_improved",
        "weight_abs_max_over_p99_improved",
        "weight_w4_sym_per_out_sqnr_db_improved",
    ]:
        summary[f"{key}_count"] = sum(str(row.get(key, "False")) == "True" or row.get(key) is True for row in rows)

    head_rows = [row for row in rows if row["head_proxy_improved"]]
    no_head_rows = [row for row in rows if not row["head_proxy_improved"]]
    for group_name, group_rows in [("head_proxy_improved", head_rows), ("head_proxy_not_improved", no_head_rows)]:
        summary[group_name] = {"count": len(group_rows)}
        for metric in [
            "activation_token_score_p99_delta",
            "activation_a4_sym_token_sqnr_db_delta",
            "weight_abs_max_over_p99_delta",
            "weight_w4_sym_per_out_sqnr_db_delta",
        ]:
            vals = [float(row[metric]) for row in group_rows if metric in row]
            summary[group_name][f"{metric}_mean"] = float(np.mean(vals)) if vals else 0.0
            summary[group_name][f"{metric}_median"] = float(np.median(vals)) if vals else 0.0

    return summary


def plot_js_heatmap(out_dir, rows, kind, left, right):
    filtered = [row for row in rows if row["kind"] == kind and row["left"] == left and row["right"] == right]
    if not filtered:
        return
    linears = REP_LINEAR_NAMES
    layers = REP_LAYERS
    grid = np.full((len(layers), len(linears)), np.nan)
    for row in filtered:
        if row["layer"] in layers and row["linear"] in linears:
            grid[layers.index(row["layer"]), linears.index(row["linear"])] = row["js"]
    plt.figure(figsize=(7, 4.8))
    plt.imshow(grid, aspect="auto", interpolation="nearest")
    plt.colorbar(label="JS divergence")
    plt.xticks(range(len(linears)), linears, rotation=45, ha="right")
    plt.yticks(range(len(layers)), layers)
    plt.xlabel("linear")
    plt.ylabel("layer")
    plt.title(f"{kind}: {left} vs {right}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"js_heatmap_{kind}_{left}_vs_{right}.png"), dpi=160)
    plt.close()


def plot_alignment_scatter(out_dir, rows):
    x = [row["head_proxy_mse_rel_delta"] for row in rows]
    y = [row.get("activation_token_score_p99_delta", 0.0) for row in rows]
    colors = ["tab:green" if row["head_proxy_improved"] else "tab:red" for row in rows]
    plt.figure(figsize=(6, 4.5))
    plt.scatter(x, y, c=colors, alpha=0.7, s=20)
    plt.axvline(0, color="black", linewidth=1, linestyle="--")
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("head_proxy_mse relative delta (SVD - baseline)")
    plt.ylabel("activation token_score_p99 delta (SVD - baseline)")
    plt.title("Head-proxy improvement vs token-outlier change")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_head_proxy_vs_token_outlier_delta.png"), dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./modelzoo/Qwen/Qwen2.5-3B")
    parser.add_argument("--baseline_flat_path", default="./outputs/Qwen2.5-3B/w4a4/qwen25_3b_base_w4a4kv4_lwc_lac_full_headmse_gpu0")
    parser.add_argument("--svd_flat_path", default="./outputs/Qwen2.5-3B/w4a4/qwen25_3b_base_w4a4kv4_lwc_lac_svd_mix_linear_a0p5_full_eval_gpu3_20260415_180145")
    parser.add_argument("--baseline_log", default="./outputs/Qwen2.5-3B/w4a4/qwen25_3b_base_w4a4kv4_lwc_lac_full_headmse_gpu0/log_rank0_20260412_114247.txt")
    parser.add_argument("--svd_log", default="./outputs/Qwen2.5-3B/w4a4/qwen25_3b_base_w4a4kv4_lwc_lac_svd_mix_linear_a0p5_full_eval_gpu3_20260415_180145/log_rank0_20260415_180148.txt")
    parser.add_argument("--prev_diag_dir", default="./outputs/diagnostics/qwen25_3b_base_outlier_dist_baseline_vs_svd_mix_linear_20260528_005938")
    parser.add_argument("--output_root", default="./outputs/diagnostics")
    parser.add_argument("--cali_dataset", default="wikitext2")
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--max_weight_samples", type=int, default=200000)
    parser.add_argument("--max_activation_samples_per_batch", type=int, default=100000)
    parser.add_argument("--max_activation_samples_total", type=int, default=200000)
    parser.add_argument("--max_token_samples_per_batch", type=int, default=50000)
    parser.add_argument("--max_token_samples_total", type=int, default=100000)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_root, f"qwen25_3b_base_kl_js_head_alignment_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    utils.seed_everything(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    data_args = SimpleNamespace(cali_dataset=args.cali_dataset)
    dataloader = data_utils.get_loaders(data_args, args.cali_dataset, tokenizer, nsamples=args.nsamples, seqlen=2048, eval_mode=False)

    all_samples = {}
    states = [("fp", None), ("baseline", args.baseline_flat_path), ("svd_mix_linear", args.svd_flat_path)]
    for state_name, flat_path in states:
        print(f"Collecting samples for {state_name}", flush=True)
        all_samples.update(collect_distribution_samples(args, state_name, flat_path, dataloader))

    divergence_rows = compute_distribution_divergences(all_samples)
    write_csv(os.path.join(out_dir, "kl_js_distribution_rows.csv"), divergence_rows)
    alignment_rows = build_alignment_rows(args)
    write_csv(os.path.join(out_dir, "head_outlier_alignment_rows.csv"), alignment_rows)

    summary = {
        "args": vars(args),
        "divergence_summary": summarize_divergences(divergence_rows),
        "alignment_summary": summarize_alignment(alignment_rows),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for kind in ["weight_abs", "activation_abs", "token_score"]:
        for left, right in [("fp", "baseline"), ("baseline", "svd_mix_linear"), ("fp", "svd_mix_linear")]:
            plot_js_heatmap(out_dir, divergence_rows, kind, left, right)
    plot_alignment_scatter(out_dir, alignment_rows)

    print(f"Saved KL/JS and alignment diagnostics to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
