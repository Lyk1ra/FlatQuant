"""Information-theoretic diagnostics for FlatQuant-transformed tensors.

Measures, per linear module and per state (fp / baseline / svd_mix_linear):

Activations (input of each linear, after the FlatQuant transform):
  - a4 SQNR under the same per-token max/7 symmetric 4-bit quantizer used in
    qwen25_outlier_distribution.py, converted to effective bits (SQNR / 6.02)
  - code entropy H(Q) of the actual 16 integer codes, in bits
  - kurtosis (Gaussian = 3, uniform = 1.8) as a distribution-shape probe
  - gap_to_gauss_db: measured SQNR minus the SQNR of the *same* quantizer on an
    iid Gaussian tensor of the same row length (simulated once per unique dim).
    A gap near 0 means the marginal distribution is already at the
    linear-transform + scalar-uniform-quantizer ceiling.
  - for representative modules: spectral entropy / effective rank of the
    channel correlation matrix and Gaussian total correlation (bits/dim),
    which quantify residual cross-channel redundancy that NO scalar quantizer
    can exploit (only vector/low-rank schemes can).

Weights:
  - w4 SQNR under per-output-channel max/7 symmetric 4-bit quantization,
    effective bits, code entropy, kurtosis, gap_to_gauss_db / gap_to_uniform_db
  - singular-value spectrum summaries: effective rank (exp of spectral
    entropy), erank ratio, stable rank.

Reference constants recorded in summary.json:
  - iid Gaussian / uniform ceilings for the same quantizer (simulated)
  - Lloyd-Max 16-level Gaussian scalar quantizer: ~20.22 dB
  - entropy-coded scalar quantizer bound (Gaussian, H=4b): ~22.55 dB
  - Shannon rate-distortion bound (Gaussian, R=4b): 24.08 dB
"""

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

DB_PER_BIT = 20.0 * math.log10(2.0)  # 6.0206 dB per bit


def fake_quant_rowmax(x2):
    """Per-row symmetric 4-bit quantization: scale = max/7, clamp [-8, 7].

    Returns (dequantized, integer codes)."""
    max_per_row = x2.abs().max(dim=1, keepdim=True)[0]
    scale = (max_per_row / 7.0).clamp(min=1e-12)
    q = torch.clamp(torch.round(x2 / scale), -8, 7)
    return q * scale, q


@torch.no_grad()
def simulate_gauss_ceiling(n, device, rows=512, seed=0):
    """SQNR and code entropy of the row-max quantizer on iid N(0,1) rows."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, n, generator=gen).to(device)
    dq, q = fake_quant_rowmax(x)
    sqnr = 10.0 * math.log10(float(x.pow(2).sum()) / max(float((dq - x).pow(2).sum()), 1e-24))
    hist = torch.histc(q.float(), bins=16, min=-8.5, max=7.5)
    p = (hist / hist.sum()).clamp(min=1e-24)
    entropy = float(-(p * torch.log2(p)).sum())
    return sqnr, entropy


@torch.no_grad()
def simulate_uniform_ceiling(n, device, rows=512, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = ((torch.rand(rows, n, generator=gen) * 2.0 - 1.0) * math.sqrt(3.0)).to(device)
    dq, q = fake_quant_rowmax(x)
    sqnr = 10.0 * math.log10(float(x.pow(2).sum()) / max(float((dq - x).pow(2).sum()), 1e-24))
    hist = torch.histc(q.float(), bins=16, min=-8.5, max=7.5)
    p = (hist / hist.sum()).clamp(min=1e-24)
    entropy = float(-(p * torch.log2(p)).sum())
    return sqnr, entropy


class InfoCollector:
    """Streaming per-module activation statistics."""

    def __init__(self, do_cov, max_cov_channels=2048, seed=0):
        self.num_values = 0
        self.sum_sq = 0.0
        self.sum_fourth = 0.0
        self.quant_err_sum = 0.0
        self.code_hist = torch.zeros(16, dtype=torch.float64)
        self.n_features = None
        self.do_cov = do_cov
        self.max_cov_channels = max_cov_channels
        self.seed = seed
        self.cov_idx = None
        self.cov_sum = None
        self.cov_mean_sum = None
        self.cov_count = 0

    @torch.no_grad()
    def update(self, x):
        x2 = x.detach().float().reshape(-1, x.shape[-1])
        if self.n_features is None:
            self.n_features = x2.shape[-1]
        self.num_values += x2.numel()
        self.sum_sq += float(x2.pow(2).sum())
        self.sum_fourth += float(x2.pow(4).sum())

        dq, q = fake_quant_rowmax(x2)
        self.quant_err_sum += float((dq - x2).pow(2).sum())
        self.code_hist += torch.histc(q.float(), bins=16, min=-8.5, max=7.5).double().cpu()

        if self.do_cov:
            if self.cov_idx is None:
                n = x2.shape[-1]
                if n > self.max_cov_channels:
                    gen = torch.Generator().manual_seed(self.seed)
                    self.cov_idx = torch.randperm(n, generator=gen)[: self.max_cov_channels].to(x2.device)
                else:
                    self.cov_idx = torch.arange(n, device=x2.device)
                d = self.cov_idx.numel()
                self.cov_sum = torch.zeros(d, d, dtype=torch.float64, device=x2.device)
                self.cov_mean_sum = torch.zeros(d, dtype=torch.float64, device=x2.device)
            xs = x2[:, self.cov_idx].double()
            self.cov_sum += xs.t() @ xs
            self.cov_mean_sum += xs.sum(dim=0)
            self.cov_count += xs.shape[0]

    def finalize(self):
        mean_sq = self.sum_sq / max(self.num_values, 1)
        kurtosis = (self.sum_fourth / max(self.num_values, 1)) / max(mean_sq ** 2, 1e-24)
        quant_mse = self.quant_err_sum / max(self.num_values, 1)
        sqnr = 10.0 * math.log10(max(self.sum_sq, 1e-24) / max(self.quant_err_sum, 1e-24))
        p = (self.code_hist / self.code_hist.sum()).clamp(min=1e-24)
        code_entropy = float(-(p * torch.log2(p)).sum())

        row = {
            "n_features": self.n_features,
            "num_values": self.num_values,
            "kurtosis_raw": kurtosis,
            "a4_sym_token_mse": quant_mse,
            "a4_sym_token_sqnr_db": sqnr,
            "a4_effective_bits": sqnr / DB_PER_BIT,
            "a4_code_entropy_bits": code_entropy,
            "a4_code_perplexity": 2.0 ** code_entropy,
        }

        if self.do_cov and self.cov_count > 1:
            d = self.cov_idx.numel()
            mean = (self.cov_mean_sum / self.cov_count).cpu()
            cov = (self.cov_sum.cpu() / self.cov_count) - torch.outer(mean, mean)
            std = cov.diag().clamp(min=1e-12).sqrt()
            corr = cov / torch.outer(std, std)
            eigvals = torch.linalg.eigvalsh(corr).clamp(min=1e-10)
            p_eig = (eigvals / eigvals.sum()).clamp(min=1e-24)
            spec_entropy_nats = float(-(p_eig * p_eig.log()).sum())
            erank = math.exp(spec_entropy_nats)
            # Gaussian total correlation of the correlation matrix, bits/dim:
            # TC = -0.5 * log2 det(R) / d  (0 => channels already independent)
            total_corr_bits = float(-0.5 * eigvals.log2().sum() / d)
            row.update({
                "cov_dim": d,
                "cov_tokens": self.cov_count,
                "spectral_entropy_bits": spec_entropy_nats / math.log(2.0),
                "erank": erank,
                "erank_ratio": erank / d,
                "total_corr_bits_per_dim": total_corr_bits,
            })
        return row


def weight_info_stats(weight, device):
    w = weight.detach().float().to(device)
    dq, q = fake_quant_rowmax(w)
    err = float((dq - w).pow(2).sum())
    sq = float(w.pow(2).sum())
    sqnr = 10.0 * math.log10(max(sq, 1e-24) / max(err, 1e-24))
    hist = torch.histc(q.float(), bins=16, min=-8.5, max=7.5)
    p = (hist / hist.sum()).clamp(min=1e-24)
    code_entropy = float(-(p * torch.log2(p)).sum())
    mean_sq = sq / w.numel()
    kurtosis = float(w.pow(4).mean()) / max(mean_sq ** 2, 1e-24)

    s = torch.linalg.svdvals(w)
    s2 = s.pow(2)
    p_s = (s2 / s2.sum()).clamp(min=1e-24)
    spec_entropy_nats = float(-(p_s * p_s.log()).sum())
    erank = math.exp(spec_entropy_nats)
    full_rank = min(w.shape)

    return {
        "in_features": w.shape[1],
        "out_features": w.shape[0],
        "kurtosis_raw": kurtosis,
        "w4_sym_per_out_sqnr_db": sqnr,
        "w4_effective_bits": sqnr / DB_PER_BIT,
        "w4_code_entropy_bits": code_entropy,
        "sv_spectral_entropy_bits": spec_entropy_nats / math.log(2.0),
        "sv_erank": erank,
        "sv_erank_ratio": erank / full_rank,
        "sv_stable_rank": float(s2.sum() / s2.max()),
    }


def make_flatquant_args(args, exp_name):
    return SimpleNamespace(
        model=args.model,
        hf_token=args.hf_token,
        seed=args.seed,
        w_bits=4, a_bits=4, q_bits=16, k_bits=4, v_bits=4,
        w_asym=False, a_asym=False, q_asym=False, k_asym=True, v_asym=True,
        w_groupsize=-1, a_groupsize=-1, q_groupsize=-1, k_groupsize=128, v_groupsize=128,
        gptq=False, gptq_mse=False, percdamp=0.01, act_order=False,
        epochs=15, cali_dataset=args.cali_dataset, nsamples=args.nsamples,
        cali_bsz=4, flat_lr=5e-3, cali_trans=True, add_diag=True, lwc=True, lac=True,
        resume=False, save_matrix=False, reload_matrix=False, matrix_path=None,
        diag_init="sq_style", diag_alpha=0.3, warmup=False,
        deactive_amp=True, direct_inv=True, separate_vtrans=False,
        output_dir=args.output_root, exp_name=exp_name,
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
def collect_state(args, state_name, flat_path, dataloader, gauss_cache, uniform_cache):
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

    def ceiling_for(n):
        if n not in gauss_cache:
            gauss_cache[n] = simulate_gauss_ceiling(n, utils.DEV)
            uniform_cache[n] = simulate_uniform_ceiling(n, utils.DEV)
        return gauss_cache[n], uniform_cache[n]

    weight_rows = []
    for name, module in target_modules.items():
        weight = module.weight if state_name == "fp" else module.linear.weight
        stats = weight_info_stats(weight, utils.DEV)
        (g_sqnr, g_ent), (u_sqnr, u_ent) = ceiling_for(stats["in_features"])
        stats.update({
            "state": state_name,
            "name": name,
            "layer": layer_index_from_name(name),
            "linear": module_short_name(name),
            "gauss_ceiling_db": g_sqnr,
            "gap_to_gauss_db": stats["w4_sym_per_out_sqnr_db"] - g_sqnr,
            "uniform_ceiling_db": u_sqnr,
            "gap_to_uniform_db": stats["w4_sym_per_out_sqnr_db"] - u_sqnr,
        })
        weight_rows.append(stats)

    collectors = {}
    for name in target_modules:
        layer_idx = layer_index_from_name(name)
        short = module_short_name(name)
        do_cov = layer_idx in REP_LAYERS and short in REP_LINEAR_NAMES
        collectors[name] = InfoCollector(do_cov=do_cov, max_cov_channels=args.max_cov_channels, seed=args.seed)

    hooks = []
    for name, module in target_modules.items():
        def hook(mod, inp, out, module_name=name):
            x = inp[0] if isinstance(inp, tuple) else inp
            collectors[module_name].update(x)
        hooks.append(module.register_forward_hook(hook))

    for idx, batch in enumerate(dataloader):
        if idx >= args.nsamples:
            break
        model(batch[0].to(utils.DEV))

    for hook in hooks:
        hook.remove()

    activation_rows = []
    for name, collector in collectors.items():
        stats = collector.finalize()
        (g_sqnr, g_ent), (u_sqnr, u_ent) = ceiling_for(stats["n_features"])
        stats.update({
            "state": state_name,
            "name": name,
            "layer": layer_index_from_name(name),
            "linear": module_short_name(name),
            "gauss_ceiling_db": g_sqnr,
            "gauss_ceiling_code_entropy_bits": g_ent,
            "gap_to_gauss_db": stats["a4_sym_token_sqnr_db"] - g_sqnr,
            "uniform_ceiling_db": u_sqnr,
            "gap_to_uniform_db": stats["a4_sym_token_sqnr_db"] - u_sqnr,
        })
        activation_rows.append(stats)

    del model
    utils.cleanup_memory(verbose=False)
    torch.cuda.empty_cache()
    return weight_rows, activation_rows


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, restval="")
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows, metric_names):
    summary = {}
    for state in sorted({row["state"] for row in rows}):
        state_rows = [row for row in rows if row["state"] == state]
        summary[state] = {}
        for metric in metric_names:
            vals = [float(row[metric]) for row in state_rows if metric in row and row[metric] != ""]
            summary[state][metric] = {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "median": float(np.median(vals)) if vals else 0.0,
                "min": float(np.min(vals)) if vals else 0.0,
                "max": float(np.max(vals)) if vals else 0.0,
            }
    return summary


def plot_heatmap(out_dir, rows, metric, kind):
    states = sorted({row["state"] for row in rows})
    linears = LINEAR_NAMES
    for state in states:
        state_rows = [row for row in rows if row["state"] == state and metric in row and row[metric] != ""]
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
    parser.add_argument("--max_cov_channels", type=int, default=2048)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_root, f"qwen25_3b_base_info_theory_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    utils.seed_everything(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    data_args = SimpleNamespace(cali_dataset=args.cali_dataset)
    dataloader = data_utils.get_loaders(data_args, args.cali_dataset, tokenizer, nsamples=args.nsamples, seqlen=2048, eval_mode=False)

    gauss_cache, uniform_cache = {}, {}
    all_weight_rows, all_activation_rows = [], []
    states = [
        ("fp", None),
        ("baseline", args.baseline_flat_path),
        ("svd_mix_linear", args.svd_flat_path),
    ]
    for state_name, flat_path in states:
        print(f"Collecting {state_name} info-theory diagnostics", flush=True)
        weight_rows, activation_rows = collect_state(args, state_name, flat_path, dataloader, gauss_cache, uniform_cache)
        all_weight_rows.extend(weight_rows)
        all_activation_rows.extend(activation_rows)
        write_csv(os.path.join(out_dir, f"weight_info_{state_name}.csv"), weight_rows)
        write_csv(os.path.join(out_dir, f"activation_info_{state_name}.csv"), activation_rows)

    write_csv(os.path.join(out_dir, "weight_info_all.csv"), all_weight_rows)
    write_csv(os.path.join(out_dir, "activation_info_all.csv"), all_activation_rows)

    summary = {
        "args": vars(args),
        "reference_constants": {
            "db_per_bit": DB_PER_BIT,
            "lloyd_max_gauss_16level_db": 20.22,
            "entropy_coded_scalar_gauss_4bit_db": 22.55,
            "shannon_rd_gauss_4bit_db": 24.08,
            "gauss_ceiling_by_dim": {str(n): {"sqnr_db": v[0], "code_entropy_bits": v[1]} for n, v in gauss_cache.items()},
            "uniform_ceiling_by_dim": {str(n): {"sqnr_db": v[0], "code_entropy_bits": v[1]} for n, v in uniform_cache.items()},
        },
        "weight_summary": summarize_rows(all_weight_rows, [
            "w4_sym_per_out_sqnr_db", "w4_effective_bits", "w4_code_entropy_bits",
            "kurtosis_raw", "gap_to_gauss_db", "gap_to_uniform_db",
            "sv_erank_ratio", "sv_stable_rank",
        ]),
        "activation_summary": summarize_rows(all_activation_rows, [
            "a4_sym_token_sqnr_db", "a4_effective_bits", "a4_code_entropy_bits",
            "kurtosis_raw", "gap_to_gauss_db", "gap_to_uniform_db",
            "erank_ratio", "total_corr_bits_per_dim",
        ]),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_heatmap(out_dir, all_activation_rows, "gap_to_gauss_db", "activation")
    plot_heatmap(out_dir, all_activation_rows, "a4_code_entropy_bits", "activation")
    plot_heatmap(out_dir, all_activation_rows, "kurtosis_raw", "activation")
    plot_heatmap(out_dir, all_weight_rows, "gap_to_gauss_db", "weight")
    plot_heatmap(out_dir, all_weight_rows, "w4_code_entropy_bits", "weight")

    print(f"Saved info-theory diagnostics to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
