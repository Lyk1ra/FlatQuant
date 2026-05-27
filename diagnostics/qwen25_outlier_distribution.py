import argparse
import csv
import json
import math
import os
import random
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
REP_LAYERS = {0, 2, 3, 20, 29, 30, 34, 35}
REP_LINEAR_NAMES = {"q_proj", "o_proj", "up_proj", "down_proj"}


class ActivationCollector:
    def __init__(self, max_abs_samples=200000, max_token_samples=50000):
        self.max_abs_samples = max_abs_samples
        self.max_token_samples = max_token_samples
        self.abs_samples = []
        self.token_samples = []
        self.channel_abs_max = None
        self.abs_max = 0.0
        self.num_values = 0
        self.sum_values = 0.0
        self.sum_sq = 0.0
        self.sum_fourth = 0.0
        self.quant_mse_sum = 0.0
        self.quant_sq_sum = 0.0

    def _append_sample(self, values, store, max_count):
        values = values.detach().flatten()
        if values.numel() == 0:
            return
        remain = max_count - sum(x.numel() for x in store)
        if remain <= 0:
            return
        take = min(remain, values.numel())
        if values.numel() > take:
            idx = torch.randperm(values.numel(), device=values.device)[:take]
            values = values[idx]
        else:
            values = values[:take]
        store.append(values.cpu())

    def update(self, x):
        x = x.detach().float()
        if x.ndim == 2:
            x2 = x.reshape(-1, x.shape[-1])
        else:
            x2 = x.reshape(-1, x.shape[-1])
        abs_x = x2.abs()

        self.abs_max = max(self.abs_max, float(abs_x.max().item()))
        self.num_values += x2.numel()
        self.sum_values += float(x2.sum().item())
        self.sum_sq += float(x2.pow(2).sum().item())
        self.sum_fourth += float(x2.pow(4).sum().item())

        channel_max = abs_x.max(dim=0)[0].cpu()
        if self.channel_abs_max is None:
            self.channel_abs_max = channel_max
        else:
            self.channel_abs_max = torch.maximum(self.channel_abs_max, channel_max)

        token_mean = abs_x.mean(dim=1).clamp(min=1e-12)
        token_score = abs_x.max(dim=1)[0] / token_mean
        self._append_sample(abs_x, self.abs_samples, self.max_abs_samples)
        self._append_sample(token_score, self.token_samples, self.max_token_samples)

        max_per_token = abs_x.max(dim=1, keepdim=True)[0]
        scale = (max_per_token / 7.0).clamp(min=1e-12)
        q = torch.clamp(torch.round(x2 / scale), -8, 7)
        dq = q * scale
        self.quant_mse_sum += float((dq - x2).pow(2).sum().item())
        self.quant_sq_sum += float(x2.pow(2).sum().item())

    def finalize(self):
        abs_samples = torch.cat(self.abs_samples).float() if self.abs_samples else torch.empty(0)
        token_samples = torch.cat(self.token_samples).float() if self.token_samples else torch.empty(0)
        channel_abs_max = self.channel_abs_max.float() if self.channel_abs_max is not None else torch.empty(0)
        mean = self.sum_values / max(self.num_values, 1)
        var = self.sum_sq / max(self.num_values, 1) - mean * mean
        std = math.sqrt(max(var, 0.0))
        fourth_mean = self.sum_fourth / max(self.num_values, 1)
        kurtosis = fourth_mean / max((self.sum_sq / max(self.num_values, 1)) ** 2, 1e-24)
        quant_mse = self.quant_mse_sum / max(self.num_values, 1)
        sqnr = 10.0 * math.log10(max(self.quant_sq_sum, 1e-24) / max(self.quant_mse_sum, 1e-24))

        return {
            "abs_max": self.abs_max,
            "mean": mean,
            "std": std,
            "kurtosis_raw": kurtosis,
            "p50_abs": quantile(abs_samples, 0.50),
            "p90_abs": quantile(abs_samples, 0.90),
            "p99_abs": quantile(abs_samples, 0.99),
            "p999_abs": quantile(abs_samples, 0.999),
            "abs_max_over_p99": self.abs_max / max(quantile(abs_samples, 0.99), 1e-12),
            "p999_over_p50": quantile(abs_samples, 0.999) / max(quantile(abs_samples, 0.50), 1e-12),
            "token_score_p50": quantile(token_samples, 0.50),
            "token_score_p99": quantile(token_samples, 0.99),
            "token_score_max_sampled": float(token_samples.max().item()) if token_samples.numel() else 0.0,
            "channel_abs_max_p50": quantile(channel_abs_max, 0.50),
            "channel_abs_max_p99": quantile(channel_abs_max, 0.99),
            "channel_abs_max_max": float(channel_abs_max.max().item()) if channel_abs_max.numel() else 0.0,
            "channel_max_over_median": (float(channel_abs_max.max().item()) / max(quantile(channel_abs_max, 0.50), 1e-12)) if channel_abs_max.numel() else 0.0,
            "a4_sym_token_mse": quant_mse,
            "a4_sym_token_sqnr_db": sqnr,
            "num_values": self.num_values,
        }

    def abs_sample_numpy(self):
        if not self.abs_samples:
            return np.array([], dtype=np.float32)
        return torch.cat(self.abs_samples).float().numpy()


def quantile(x, q):
    if x.numel() == 0:
        return 0.0
    return float(torch.quantile(x.float(), q).item())


def tensor_abs_sample(x, max_count):
    x = x.detach().float().flatten()
    if x.numel() <= max_count:
        return x.abs().cpu()
    idx = torch.randperm(x.numel(), device=x.device)[:max_count]
    return x[idx].abs().cpu()


def weight_stats(weight, max_quantile_values=2000000):
    w = weight.detach().float().cpu()
    abs_w = w.abs().flatten()
    if abs_w.numel() > max_quantile_values:
        idx = torch.randperm(abs_w.numel())[:max_quantile_values]
        abs_w_q = abs_w[idx]
    else:
        abs_w_q = abs_w
    per_out_abs_max = w.abs().amax(dim=1)
    max_per_out = per_out_abs_max.reshape(-1, 1).clamp(min=1e-12)
    scale = max_per_out / 7.0
    q = torch.clamp(torch.round(w / scale), -8, 7)
    dq = q * scale
    mse = (dq - w).pow(2).mean().item()
    sq = w.pow(2).mean().item()
    return {
        "abs_max": float(abs_w.max().item()),
        "mean_abs": float(abs_w.mean().item()),
        "std": float(w.std().item()),
        "kurtosis_raw": float(w.pow(4).mean().item() / max(w.pow(2).mean().item() ** 2, 1e-24)),
        "p50_abs": quantile(abs_w_q, 0.50),
        "p90_abs": quantile(abs_w_q, 0.90),
        "p99_abs": quantile(abs_w_q, 0.99),
        "p999_abs": quantile(abs_w_q, 0.999),
        "abs_max_over_p99": float(abs_w.max().item()) / max(quantile(abs_w_q, 0.99), 1e-12),
        "p999_over_p50": quantile(abs_w_q, 0.999) / max(quantile(abs_w_q, 0.50), 1e-12),
        "per_out_abs_max_p50": quantile(per_out_abs_max, 0.50),
        "per_out_abs_max_p99": quantile(per_out_abs_max, 0.99),
        "per_out_max_over_median": float(per_out_abs_max.max().item()) / max(quantile(per_out_abs_max, 0.50), 1e-12),
        "w4_sym_per_out_mse": mse,
        "w4_sym_per_out_sqnr_db": 10.0 * math.log10(max(sq, 1e-24) / max(mse, 1e-24)),
        "num_values": w.numel(),
    }


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
    for i, part in enumerate(parts[:-1]):
        if part == "layers":
            return int(parts[i + 1])
    return -1


def get_target_modules(model, state_name):
    targets = {}
    if state_name == "fp":
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name != "lm_head" and any(f".{x}" in name for x in LINEAR_NAMES):
                targets[name] = module
    else:
        for name, module in model.named_modules():
            if isinstance(module, FlatQuantizedLinear) and any(name.endswith(f".{x}") for x in LINEAR_NAMES):
                targets[name] = module
    return targets


@torch.no_grad()
def collect_state(args, state_name, flat_path, dataloader, tokenizer):
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

    target_modules = get_target_modules(model, state_name)
    act_collectors = {name: ActivationCollector(args.max_abs_samples, args.max_token_samples) for name in target_modules}
    weight_rows = []
    hist_samples = {}

    for name, module in target_modules.items():
        weight = module.weight if state_name == "fp" else module.linear.weight
        stats = weight_stats(weight)
        stats.update({
            "state": state_name,
            "name": name,
            "layer": layer_index_from_name(name),
            "linear": module_short_name(name),
        })
        weight_rows.append(stats)
        layer_idx = stats["layer"]
        short = stats["linear"]
        if layer_idx in REP_LAYERS and short in REP_LINEAR_NAMES:
            hist_samples[(state_name, name, "weight")] = tensor_abs_sample(weight, args.max_hist_samples).numpy()

    hooks = []
    for name, module in target_modules.items():
        def hook(mod, inp, out, module_name=name):
            x = inp[0] if isinstance(inp, tuple) else inp
            act_collectors[module_name].update(x)
        hooks.append(module.register_forward_hook(hook))

    for idx, batch in enumerate(dataloader):
        if idx >= args.nsamples:
            break
        model(batch[0].to(utils.DEV))

    for hook in hooks:
        hook.remove()

    activation_rows = []
    for name, collector in act_collectors.items():
        stats = collector.finalize()
        stats.update({
            "state": state_name,
            "name": name,
            "layer": layer_index_from_name(name),
            "linear": module_short_name(name),
        })
        activation_rows.append(stats)
        if stats["layer"] in REP_LAYERS and stats["linear"] in REP_LINEAR_NAMES:
            hist_samples[(state_name, name, "activation")] = collector.abs_sample_numpy()

    del model
    utils.cleanup_memory(verbose=False)
    torch.cuda.empty_cache()
    return weight_rows, activation_rows, hist_samples


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows, metric_names):
    summary = {}
    for state in sorted({row["state"] for row in rows}):
        state_rows = [row for row in rows if row["state"] == state]
        summary[state] = {}
        for metric in metric_names:
            vals = [float(row[metric]) for row in state_rows if metric in row]
            summary[state][metric] = {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "median": float(np.median(vals)) if vals else 0.0,
                "max": float(np.max(vals)) if vals else 0.0,
            }
    return summary


def plot_histograms(out_dir, hist_samples):
    grouped = {}
    for (state, name, kind), values in hist_samples.items():
        grouped.setdefault((name, kind), {})[state] = values
    hist_dir = os.path.join(out_dir, "histograms")
    os.makedirs(hist_dir, exist_ok=True)
    for (name, kind), state_values in grouped.items():
        if len(state_values) < 2:
            continue
        plt.figure(figsize=(7, 4))
        for state in ["fp", "baseline", "svd_mix_linear"]:
            values = state_values.get(state)
            if values is None or len(values) == 0:
                continue
            upper = np.quantile(values, 0.999)
            clipped = values[values <= upper]
            plt.hist(clipped, bins=120, density=True, histtype="step", linewidth=1.5, label=state)
        plt.title(f"{kind}: {name}")
        plt.xlabel("abs(value), clipped at p99.9 for display")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        safe_name = name.replace(".", "_")
        plt.savefig(os.path.join(hist_dir, f"{kind}_{safe_name}.png"), dpi=160)
        plt.close()


def plot_heatmap(out_dir, rows, metric, kind):
    states = ["fp", "baseline", "svd_mix_linear"]
    linears = LINEAR_NAMES
    for state in states:
        state_rows = [row for row in rows if row["state"] == state]
        if not state_rows:
            continue
        num_layers = max(row["layer"] for row in state_rows) + 1
        grid = np.full((num_layers, len(linears)), np.nan)
        for row in state_rows:
            if row["linear"] in linears:
                grid[row["layer"], linears.index(row["linear"])] = float(row[metric])
        plt.figure(figsize=(9, 7))
        plt.imshow(grid, aspect="auto", interpolation="nearest")
        plt.colorbar(label=metric)
        plt.xticks(range(len(linears)), linears, rotation=45, ha="right")
        plt.yticks(range(num_layers), range(num_layers))
        plt.title(f"{kind} {metric}: {state}")
        plt.xlabel("linear")
        plt.ylabel("layer")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"heatmap_{kind}_{metric}_{state}.png"), dpi=160)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./modelzoo/Qwen/Qwen2.5-3B")
    parser.add_argument("--baseline_flat_path", default="./outputs/Qwen2.5-3B/w4a4/qwen25_3b_base_w4a4kv4_lwc_lac_full_headmse_gpu0")
    parser.add_argument("--svd_flat_path", default="./outputs/Qwen2.5-3B/w4a4/qwen25_3b_base_w4a4kv4_lwc_lac_svd_mix_linear_a0p5_full_eval_gpu3_20260415_180145")
    parser.add_argument("--output_root", default="./outputs/diagnostics")
    parser.add_argument("--cali_dataset", default="wikitext2")
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--max_abs_samples", type=int, default=200000)
    parser.add_argument("--max_token_samples", type=int, default=50000)
    parser.add_argument("--max_hist_samples", type=int, default=200000)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_root, f"qwen25_3b_base_outlier_dist_baseline_vs_svd_mix_linear_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    utils.seed_everything(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    data_args = SimpleNamespace(cali_dataset=args.cali_dataset)
    dataloader = data_utils.get_loaders(data_args, args.cali_dataset, tokenizer, nsamples=args.nsamples, seqlen=2048, eval_mode=False)

    all_weight_rows = []
    all_activation_rows = []
    all_hist_samples = {}
    states = [
        ("fp", None),
        ("baseline", args.baseline_flat_path),
        ("svd_mix_linear", args.svd_flat_path),
    ]
    for state_name, flat_path in states:
        print(f"Collecting {state_name} diagnostics", flush=True)
        weight_rows, activation_rows, hist_samples = collect_state(args, state_name, flat_path, dataloader, tokenizer)
        all_weight_rows.extend(weight_rows)
        all_activation_rows.extend(activation_rows)
        all_hist_samples.update(hist_samples)
        write_csv(os.path.join(out_dir, f"weight_stats_after_{state_name}.csv"), weight_rows)
        write_csv(os.path.join(out_dir, f"activation_stats_after_{state_name}.csv"), activation_rows)

    write_csv(os.path.join(out_dir, "weight_stats_all.csv"), all_weight_rows)
    write_csv(os.path.join(out_dir, "activation_stats_all.csv"), all_activation_rows)

    summary = {
        "args": vars(args),
        "weight_summary": summarize_rows(all_weight_rows, ["abs_max_over_p99", "p999_over_p50", "per_out_max_over_median", "w4_sym_per_out_mse", "w4_sym_per_out_sqnr_db"]),
        "activation_summary": summarize_rows(all_activation_rows, ["abs_max_over_p99", "p999_over_p50", "token_score_p99", "channel_max_over_median", "a4_sym_token_mse", "a4_sym_token_sqnr_db"]),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_histograms(out_dir, all_hist_samples)
    plot_heatmap(out_dir, all_weight_rows, "abs_max_over_p99", "weight")
    plot_heatmap(out_dir, all_weight_rows, "w4_sym_per_out_sqnr_db", "weight")
    plot_heatmap(out_dir, all_activation_rows, "token_score_p99", "activation")
    plot_heatmap(out_dir, all_activation_rows, "a4_sym_token_sqnr_db", "activation")

    print(f"Saved diagnostics to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
