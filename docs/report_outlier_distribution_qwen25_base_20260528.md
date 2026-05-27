# Qwen2.5-3B Base Outlier Distribution Diagnostic Report

## 1. Motivation

This diagnostic round was added after the SVD / A+B runs showed only small changes in final PPL and zero-shot accuracy.

The question was not whether the final metric changed, but whether the method changed the quantization-relevant distributions in the intended way:

- whether weight outliers became less severe
- whether token / activation outliers became less severe
- whether the transformed tensors became more quantization-friendly
- whether A+B SVD-FlatQuant has an additional distribution-level effect beyond the established FlatQuant baseline

This follows the teacher's suggestion that a new quantization method should be analyzed from the distribution side, not only from final metrics. In particular, the analysis should look at the tensors before and after the method, use both numeric metrics and visual plots, and inspect whether the distribution becomes more uniform / less outlier-dominated.

## 2. Teacher Comments Extracted As Experimental Requirements

The useful experimental requirements extracted from the comments were:

1. Compare before and after the transformation.
2. Check whether adding SVD actually affects weight / activation distributions, because final metrics alone may hide a very small internal effect.
3. Use multi-scale visualizations, not only one scalar result.
4. Use concrete metrics in addition to plots.
5. Include signal-to-noise style quantization metrics.
6. Consider KL-style distribution comparison as a future extension.
7. Check whether the SVD method preserves the semantically more important part by reducing the corresponding loss.

This report covers requirements 1-5 directly and connects requirement 7 to the existing `head_proxy_mse` diagnostics. KL-style distribution comparison is listed as follow-up work because the current run used histogram / quantile / SQNR diagnostics rather than KL divergence.

## 3. Compared States

The diagnostic compared three states on `Qwen2.5-3B` base:

| State | Meaning | Source |
| --- | --- | --- |
| `fp` | original floating-point model before FlatQuant transformation | `./modelzoo/Qwen/Qwen2.5-3B` |
| `baseline` | established normal-scale FlatQuant baseline | `outputs/Qwen2.5-3B/w4a4/qwen25_3b_base_w4a4kv4_lwc_lac_full_headmse_gpu0/` |
| `svd_mix_linear` | A+B SVD-FlatQuant, mixed SVD loss with linear layer schedule | `outputs/Qwen2.5-3B/w4a4/qwen25_3b_base_w4a4kv4_lwc_lac_svd_mix_linear_a0p5_full_eval_gpu3_20260415_180145/` |

The diagnostic reused existing `flat_parameters.pth` files. It did not rerun 15-epoch calibration.

## 4. Diagnostic Script And Output

Script:

- `diagnostics/qwen25_outlier_distribution.py`

Run environment:

- conda environment: `flatquant_svd`
- GPU: `0`

Command:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n flatquant_svd python diagnostics/qwen25_outlier_distribution.py --nsamples 4
```

Output directory:

- `outputs/diagnostics/qwen25_3b_base_outlier_dist_baseline_vs_svd_mix_linear_20260528_005938/`

Main output files:

- `summary.json`
- `weight_stats_all.csv`
- `activation_stats_all.csv`
- per-state CSV files
- heatmaps for weight / activation diagnostics
- representative weight / activation histograms

## 5. Scope Of Collected Tensors

For each transformer layer, the diagnostic collected statistics for:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

Representative histogram layers:

- layer `0`
- layer `2`
- layer `3`
- layer `20`
- layer `29`
- layer `30`
- layer `34`
- layer `35`

These representative layers match the earlier `head_proxy_mse` diagnostic style and include both early, middle, and late layers.

## 6. Metrics

### 6.1 Weight Metrics

For each linear weight tensor:

- `abs_max`
- `mean_abs`
- `std`
- `p50_abs`
- `p90_abs`
- `p99_abs`
- `p999_abs`
- `abs_max_over_p99`
- `p999_over_p50`
- `per_out_max_over_median`
- `w4_sym_per_out_mse`
- `w4_sym_per_out_sqnr_db`

`w4_sym_per_out_sqnr_db` is the signal-to-noise style metric for per-output-channel symmetric W4 fake quantization.

### 6.2 Activation / Token Metrics

For each linear module input activation:

- `abs_max`
- `p50_abs`
- `p90_abs`
- `p99_abs`
- `p999_abs`
- `abs_max_over_p99`
- `p999_over_p50`
- `token_score_p99`
- `channel_max_over_median`
- `a4_sym_token_mse`
- `a4_sym_token_sqnr_db`

The token outlier score is:

```text
max(abs(token)) / mean(abs(token))
```

`a4_sym_token_sqnr_db` is the signal-to-noise style metric for per-token symmetric A4 fake quantization.

## 7. Global Summary

### 7.1 Weight Summary

| Metric | FP | Baseline | SVD A+B |
| --- | ---: | ---: | ---: |
| `abs_max_over_p99` mean | `6.6690` | `2.6596` | `2.5746` |
| `p999_over_p50` mean | `10.0473` | `5.6902` | `5.6919` |
| `per_out_max_over_median` mean | `5.0389` | `3.0766` | `2.9404` |
| `w4_sym_per_out_mse` mean | `2.2760e-05` | `9.6427e-05` | `9.7644e-05` |
| `w4_sym_per_out_sqnr_db` mean | `14.9394` | `19.4555` | `19.4392` |

### 7.2 Activation Summary

| Metric | FP | Baseline | SVD A+B |
| --- | ---: | ---: | ---: |
| `abs_max_over_p99` mean | `218.4837` | `2.9569` | `2.9881` |
| `p999_over_p50` mean | `58.5018` | `5.8683` | `5.8460` |
| `token_score_p99` mean | `119.0528` | `5.8646` | `5.8688` |
| `channel_max_over_median` mean | `6511.8537` | `1.6619` | `1.6744` |
| `a4_sym_token_mse` mean | `0.155963` | `0.000833` | `0.000820` |
| `a4_sym_token_sqnr_db` mean | `6.4133` | `16.3468` | `16.3404` |

## 8. Baseline vs SVD A+B Improvement Counts

The following counts compare SVD A+B against the baseline across all `252` layer-linear modules.

For outlier ratios and MSE, smaller is better. For SQNR, larger is better.

### 8.1 Weight Counts

| Metric | SVD A+B Improved / Total | Mean Delta `(SVD - Baseline)` | Median Delta |
| --- | ---: | ---: | ---: |
| `abs_max_over_p99` | `146 / 252` | `-0.08501374` | `-0.01417280` |
| `per_out_max_over_median` | `146 / 252` | `-0.13618744` | `-0.01849365` |
| `w4_sym_per_out_sqnr_db` | `66 / 252` | `-0.01631684` | `-0.01040521` |

### 8.2 Activation Counts

| Metric | SVD A+B Improved / Total | Mean Delta `(SVD - Baseline)` | Median Delta |
| --- | ---: | ---: | ---: |
| `token_score_p99` | `106 / 252` | `0.00415793` | `0.00479412` |
| `channel_max_over_median` | `133 / 252` | `0.01247359` | `-0.01031278` |
| `a4_sym_token_sqnr_db` | `94 / 252` | `-0.00639549` | `-0.00396397` |
| `a4_sym_token_mse` | `127 / 252` | `-0.00001257` | approximately `0` |

These counts show that SVD A+B changes the distribution locally, but it does not produce a systematic global distribution improvement over the established FlatQuant baseline.

## 9. Representative Late-Layer Comparisons

Late layers are important because the earlier `head_proxy_mse` diagnostic showed that SVD methods can improve output-head-sensitive geometry there.

### 9.1 Activation: Layer 34 `o_proj`

| Metric | FP | Baseline | SVD A+B |
| --- | ---: | ---: | ---: |
| `token_score_p99` | `24.683493` | `7.686549` | `7.694026` |
| `channel_max_over_median` | `2.573427` | `2.275362` | `2.062016` |
| `a4_sym_token_sqnr_db` | `9.562100` | `15.444544` | `15.357902` |
| `a4_sym_token_mse` | `0.069418` | `0.012709` | `0.011612` |
| `abs_max_over_p99` | `4.460606` | `4.651852` | `4.222222` |

### 9.2 Activation: Layer 35 `down_proj`

| Metric | FP | Baseline | SVD A+B |
| --- | ---: | ---: | ---: |
| `token_score_p99` | `470.694580` | `6.062948` | `6.152538` |
| `channel_max_over_median` | `132.961240` | `1.952569` | `1.966387` |
| `a4_sym_token_sqnr_db` | `8.638284` | `15.602676` | `15.575827` |
| `a4_sym_token_mse` | `0.411832` | `0.000892` | `0.000808` |
| `abs_max_over_p99` | `463.567568` | `5.678161` | `5.742331` |

### 9.3 Weight: Layer 35 `down_proj`

| Metric | FP | Baseline | SVD A+B |
| --- | ---: | ---: | ---: |
| `abs_max_over_p99` | `21.372263` | `4.275304` | `3.801527` |
| `per_out_max_over_median` | `13.130045` | `4.327869` | `3.860465` |
| `w4_sym_per_out_sqnr_db` | `13.073733` | `19.885132` | `19.825805` |
| `w4_sym_per_out_mse` | `0.000030` | `0.000104` | `0.000116` |

## 10. Figures

The diagnostic generated heatmaps and histograms under:

- `outputs/diagnostics/qwen25_3b_base_outlier_dist_baseline_vs_svd_mix_linear_20260528_005938/`

### 10.1 Activation Token Outlier Heatmaps

FP:

![FP activation token score heatmap](assets/qwen25_outlier_heatmap_activation_token_score_p99_fp.png)

Baseline:

![Baseline activation token score heatmap](assets/qwen25_outlier_heatmap_activation_token_score_p99_baseline.png)

SVD A+B:

![SVD activation token score heatmap](assets/qwen25_outlier_heatmap_activation_token_score_p99_svd_mix_linear.png)

### 10.2 Activation A4 SQNR Heatmaps

FP:

![FP activation A4 SQNR heatmap](assets/qwen25_outlier_heatmap_activation_a4_sym_token_sqnr_db_fp.png)

Baseline:

![Baseline activation A4 SQNR heatmap](assets/qwen25_outlier_heatmap_activation_a4_sym_token_sqnr_db_baseline.png)

SVD A+B:

![SVD activation A4 SQNR heatmap](assets/qwen25_outlier_heatmap_activation_a4_sym_token_sqnr_db_svd_mix_linear.png)

### 10.3 Weight Outlier Heatmaps

FP:

![FP weight outlier heatmap](assets/qwen25_outlier_heatmap_weight_abs_max_over_p99_fp.png)

Baseline:

![Baseline weight outlier heatmap](assets/qwen25_outlier_heatmap_weight_abs_max_over_p99_baseline.png)

SVD A+B:

![SVD weight outlier heatmap](assets/qwen25_outlier_heatmap_weight_abs_max_over_p99_svd_mix_linear.png)

### 10.4 Representative Histograms

Layer 35 `down_proj` activation:

![Layer 35 down_proj activation histogram](assets/qwen25_outlier_hist_activation_layer35_down_proj.png)

Layer 35 `down_proj` weight:

![Layer 35 down_proj weight histogram](assets/qwen25_outlier_hist_weight_layer35_down_proj.png)

Layer 34 `o_proj` activation:

![Layer 34 o_proj activation histogram](assets/qwen25_outlier_hist_activation_layer34_o_proj.png)

Layer 34 `o_proj` weight:

![Layer 34 o_proj weight histogram](assets/qwen25_outlier_hist_weight_layer34_o_proj.png)

## 11. Interpretation

### 11.1 Baseline FlatQuant Has A Strong Distribution Effect

The FP activation distributions contain very large token/channel outliers. For example, the activation summary shows:

- FP `token_score_p99` mean: `119.0528`
- baseline `token_score_p99` mean: `5.8646`
- FP `a4_sym_token_sqnr_db` mean: `6.4133`
- baseline `a4_sym_token_sqnr_db` mean: `16.3468`

This means the established FlatQuant baseline already performs the expected quantization transformation: it greatly reduces token outliers and makes activation distributions much more A4-friendly.

### 11.2 SVD A+B Does Not Add A Strong New Distribution-Flattening Effect

Compared with the baseline, SVD A+B does not consistently improve token outlier scores or A4 SQNR:

- baseline `token_score_p99` mean: `5.8646`
- SVD A+B `token_score_p99` mean: `5.8688`
- baseline `a4_sym_token_sqnr_db` mean: `16.3468`
- SVD A+B `a4_sym_token_sqnr_db` mean: `16.3404`

For weights, SVD A+B slightly reduces some outlier ratios, but this does not translate into better W4 SQNR:

- baseline `abs_max_over_p99` mean: `2.6596`
- SVD A+B `abs_max_over_p99` mean: `2.5746`
- baseline `w4_sym_per_out_sqnr_db` mean: `19.4555`
- SVD A+B `w4_sym_per_out_sqnr_db` mean: `19.4392`

So the A+B method has local distribution effects, but it is not a strong global distribution-uniformization method.

### 11.3 Connection To `head_proxy_mse`

The earlier `head_proxy_mse` diagnostic showed that SVD methods can improve output-head-sensitive geometry in several layers, especially late layers.

This distribution diagnostic shows a different but compatible result:

- SVD A+B can change the error geometry
- but it does not clearly make the ordinary weight / activation distributions more quantization-friendly than the baseline

Therefore, the current SVD A+B method should be understood mainly as an error-geometry reweighting method, not as an additional outlier-removal or distribution-flattening method.

This also explains why the final PPL and zero-shot gains are limited: the method affects semantically motivated directions, but it does not give a strong distribution-level quantization advantage over the established FlatQuant baseline.

## 12. Conclusion

The mechanism-level conclusion from this round is:

1. The established FlatQuant baseline strongly reduces outlier-dominated FP activation distributions and makes them much more quantization-friendly.
2. The current A+B SVD method does not provide a clear additional global distribution-flattening effect over that baseline.
3. A+B SVD's current benefit is better described as changing output-head-sensitive error geometry, consistent with the previous `head_proxy_mse` findings.
4. The lack of a strong distribution improvement is a plausible reason why A+B does not produce a clean across-the-board PPL improvement.

## 13. Follow-Up Experiments

The next diagnostics should extend this report in two directions.

### 13.1 KL-Style Distribution Comparison

The teacher suggested a distillation-like KL analysis. A practical next version can compute histogram KL / JS divergence between:

- FP and baseline
- baseline and SVD A+B
- FP and SVD A+B

Possible distributions:

- normalized absolute activation histograms
- token outlier score histograms
- channel max histograms
- quantization error histograms

JS divergence may be preferable to raw KL because it is symmetric and more numerically stable for histogram comparisons.

### 13.2 Semantic / Head-Sensitive Loss Alignment

To connect distribution analysis with the SVD motivation, future reports should jointly plot:

- `plain_mse`
- `head_proxy_mse`
- token outlier metrics
- A4/W4 SQNR

for the same layers.

This would directly test whether layers with improved `head_proxy_mse` also show any distribution-side change, or whether the improvement is purely geometric in the output-head-induced metric.
