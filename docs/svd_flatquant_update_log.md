# SVD-Weighted FlatQuant Update Log

## Recording Requirements

1. When writing experiment reports in this log, do not write summary-style or conclusion-style statements in the experiment-result section; record the actual experimental data directly.
2. When launching experiments, include a timestamp in the experiment naming/logging path to avoid ambiguity across reruns.
3. Do not include `third-party/cutlass` in routine experiment-related commits for this project.

## Fixed Normal-Scale Protocol

For `Qwen2.5-3B` base experiments in this project, the default and continuing experiment scale is the established normal-scale configuration:

- `nsamples=128`
- `epochs=15`
- `cali_bsz=4`
- `flat_lr=5e-3`
- `W4A4KV4`
- `--cali_trans --add_diag --lwc --lac --deactive_amp --direct_inv`

Unless explicitly recorded as an exceptional diagnostic run, we should continue to use this normal-scale configuration for baseline and new method comparisons.

This means:

- new baseline-vs-method comparisons should stay on the normal scale above
- reduced-budget screening runs should not be treated as the working default for this project
- if any non-normal-scale run is ever necessary, it must be clearly marked as exceptional and must not be confused with the normal-scale comparison track

## 2026-04-10

### Goal

Integrate the `proj/SVD_A` singular value analysis into FlatQuant and compare:

- baseline FlatQuant
- SVD-weighted FlatQuant

using Qwen2.5-3B-Instruct with WikiText2 and C4 perplexity evaluation.

### Environment Work

- Kept the original `flatquant` conda environment unchanged.
- Created a new environment: `flatquant_svd`.
- Installed a Qwen2-compatible stack in `flatquant_svd`:
  - `torch 2.3.1+cu121`
  - `transformers 4.45.0`

This separation was necessary because directly upgrading the old environment could break the previously verified LLaMA3-8B workflow.

### Code Changes

#### 1. Qwen2.5 model path support

Updated FlatQuant model-name matching so local Qwen2.5 model paths are correctly recognized.

Affected file:

- `flatquant/model_utils.py`

#### 2. Added SVD-loss CLI arguments

Added the following arguments:

- `--svd_loss`
- `--svd_file`
- `--svd_weight_mode`

Affected file:

- `flatquant/args_utils.py`

#### 3. Implemented SVD-weighted reconstruction loss

Added logic to:

- load `s` and `Vh` from the `.npz` file produced by `proj/SVD_A`
- construct `V = Vh.T`
- construct directional weights from singular values
- replace plain MSE with an SVD-weighted reconstruction loss when `--svd_loss` is enabled

Affected file:

- `flatquant/train_utils.py`

#### 4. Added CUDA linalg backend fallback attempt

Added a `preferred_linalg_library("magma")` fallback attempt to reduce cuSOLVER-related failures observed during Qwen runs.

Affected file:

- `flatquant/trans_utils.py`

### Mathematical Meaning

The weighted reconstruction loss is based on the SVD of the output head:

- `W = lm_head.weight = U diag(s) V^T`

For hidden-state reconstruction error:

- `e = y_fp - y_quant`

we use:

- `e_proj = e @ V`
- `loss = mean(e_proj^2 * w)`

with:

- `w = s^2 / mean(s^2)`

This corresponds to using the quadratic form induced by:

- `W^T W = V diag(s^2) V^T`

Interpretation:

- errors along directions that are more sensitive for the final logits should receive larger penalties
- this is intended to better match the error-propagation view than uniform hidden-space MSE

### Stability Investigation

Qwen2.5-3B was not immediately stable under the original stronger training settings.

Observed issues during debugging:

- Layer 0 produced `nan` when `lwc` and `lac` were enabled
- AMP caused backward/autocast failures on Layer 0
- larger runs could terminate early without useful buffered logs

The stable configuration found during debugging was:

- `--deactive_amp`
- disable `--lwc`
- disable `--lac`
- keep `--cali_trans`
- keep `--add_diag`

### First Complete Comparison Run

To guarantee a complete, non-truncated comparison, we ran both methods with the same small stable setup:

- model: `Qwen2.5-3B-Instruct`
- `nsamples=8`
- `epochs=2`
- `cali_bsz=2`
- `--cali_trans --add_diag --deactive_amp`
- no `lwc/lac`

The only difference between the two runs was whether SVD-weighted loss was enabled.

### Results

#### Baseline FlatQuant

- WikiText2 PPL: `23.8556`
- C4 PPL: `41.3417`

#### SVD-weighted FlatQuant

- WikiText2 PPL: `25.5558`
- C4 PPL: `47.4364`

### Result Interpretation

Under the current first implementation and this small stable setting:

- SVD-weighted FlatQuant performed worse than the baseline on both WikiText2 and C4.

This does **not** necessarily invalidate the idea. It only shows that the current implementation/configuration is not yet beneficial.

Likely reasons include:

- the weighting `s^2 / mean(s^2)` may still be too strong
- using only `lm_head` spectrum for all layers may be too coarse
- the current run is small (`nsamples=8`, `epochs=2`)
- the stable Qwen setup required disabling `lwc/lac`, which differs from the stronger original FlatQuant regime

### Recommended Next Improvements

Suggested next directions:

1. Try weaker weighting functions
   - for example use `sigma` instead of `sigma^2`
   - or a softer normalized weighting

2. Try mixed loss instead of full replacement
   - `loss = alpha * mse + beta * weighted_loss`

3. Re-test on a moderately larger but still stable setup
   - for example `nsamples=16`, `epochs=4`

4. Revisit whether layer-specific spectral information is needed instead of a single shared `lm_head` spectrum

### Maintenance Note

This file is intended to be updated continuously as further SVD-FlatQuant experiments and code changes are made.

## 2026-04-11

### Goal Update

After confirming that the previous Qwen2.5-3B-Instruct SVD comparison was performed only under a small, weakened stability configuration, we switched the main experimental target to:

- `Qwen2.5-3B` **base**

The new objective was:

1. establish a reliable full FlatQuant baseline on the base model
2. re-compute the LM-head SVD for the base model rather than reusing the instruct-model spectrum
3. compare the stable full baseline against the SVD-weighted variant under the same full configuration

### Why We Switched to the Base Model

We decided to use the **base** model instead of the instruct model for the following reason:

- the downstream target is quantization methodology research rather than instruction tuning evaluation
- a base model is therefore the cleaner object for subsequent structural modifications such as SVD-based loss design

### Stable Full FlatQuant Baseline Search on Qwen2.5-3B Base

We first evaluated the unquantized floating-point model to establish a reference:

#### FP baseline

- WikiText2 PPL: `8.0357`
- C4 PPL: `13.3594`

We then tested a sequence of FlatQuant configurations under the full paper-style calibration regime:

- `nsamples=128`
- `epochs=15`
- `cali_bsz=4`
- `flat_lr=5e-3`
- `W4A4KV4`
- `KV group size = 128`
- `--cali_trans --add_diag`
- `--deactive_amp --direct_inv`

#### 1. No-clip strong baseline

Configuration:

- no `lwc`
- no `lac`

Results:

- WikiText2 PPL: `9.7057`
- C4 PPL: `15.7275`

This confirmed that Qwen2.5-3B base can be quantized stably under the full training budget even without clipping parameters.

#### 2. Only `lwc`

Results:

- WikiText2 PPL: `9.1693`
- C4 PPL: `15.1297`

This showed that `lwc` alone is stable and improves over the no-clip baseline.

#### 3. Only `lac`

Results:

- WikiText2 PPL: `9.2173`
- C4 PPL: `15.0455`

This showed that `lac` alone is also stable and improves over the no-clip baseline.

#### 4. Full baseline with `lwc + lac`

Results:

- WikiText2 PPL: `8.8095`
- C4 PPL: `14.5661`

This is the best FlatQuant baseline we obtained on Qwen2.5-3B base in the current codebase, and it became the reference baseline for the subsequent SVD experiment.

### Interpretation of the Baseline Search

The earlier very poor results previously observed on Qwen should not be interpreted as evidence that FlatQuant fundamentally fails on Qwen2.5-3B. Under the full paper-style budget and the stabilized engineering settings:

- `--deactive_amp`
- `--direct_inv`

the model can be quantized normally.

Moreover:

- `lwc` alone helps
- `lac` alone helps
- `lwc + lac` helps the most

So the working conclusion is that the stable full FlatQuant baseline on Qwen2.5-3B base is valid and usable for further research.

### Recomputing SVD for the Base Model

Because the previous SVD analysis file had been generated from the instruct model, we recomputed the LM-head SVD specifically for the base model.

Source project:

- `proj/SVD_A`

Adjustment made:

- switched `SVD_A/config.py` from `Qwen/Qwen2.5-3B-Instruct` to the local base-model path

Final model path used for SVD computation:

- `/gammadisk/liuxuanang/proj/FlatQuant/modelzoo/Qwen/Qwen2.5-3B`

LM-head matrix shape:

- `(151936, 2048)`

Generated SVD file:

- `/gammadisk/liuxuanang/proj/SVD_A/results/svd/_gammadisk_liuxuanang_proj_FlatQuant_modelzoo_Qwen_Qwen2.5-3B_svd.npz`

### Full SVD-Weighted FlatQuant Run on Qwen2.5-3B Base

We then ran the SVD-weighted variant under the same full, stable baseline configuration, with the additional arguments:

- `--svd_loss`
- `--svd_file /gammadisk/liuxuanang/proj/SVD_A/results/svd/_gammadisk_liuxuanang_proj_FlatQuant_modelzoo_Qwen_Qwen2.5-3B_svd.npz`
- `--svd_weight_mode sigma2_norm`

Results:

- WikiText2 PPL: `8.8400`
- C4 PPL: `14.5785`

### Comparison: Best Baseline vs SVD-Weighted Version

#### Best baseline (`lwc + lac`)

- WikiText2 PPL: `8.8095`
- C4 PPL: `14.5661`

#### SVD-weighted baseline (`lwc + lac + svd_loss`)

- WikiText2 PPL: `8.8400`
- C4 PPL: `14.5785`

### Final Interpretation of Current SVD Attempt

The current SVD integration is **functionally correct and numerically stable** on Qwen2.5-3B base:

- the base-specific SVD file is correctly loaded
- the run completes successfully
- there is no NaN or OOM failure

However, under the current formulation:

- shared LM-head spectrum across all layers
- `sigma2_norm` directional weighting
- full replacement of plain MSE by SVD-weighted reconstruction loss

the SVD-weighted variant does **not** outperform the best stable baseline. It is only slightly worse on both WikiText2 and C4.

This means the present conclusion is:

- **the SVD feature has been successfully integrated into FlatQuant for Qwen2.5-3B base**
- **but the current weighting design does not yet provide empirical gain over the stable non-SVD baseline**

### Recommended Next Steps

The next SVD-specific improvements should focus on the loss design rather than baseline stability. Reasonable directions include:

1. try weaker weighting than `sigma^2`
2. add a mixed objective such as `alpha * mse + beta * svd_loss`
3. test alternative normalized weight functions
4. investigate whether a single shared LM-head spectrum is too coarse for all transformer layers

## 2026-04-12

### Diagnostic Goal

After obtaining the result that the SVD-weighted run was slightly worse in final PPL than the best non-SVD baseline, we investigated a more precise question:

- does the SVD-weighted method at least reduce the error that is more directly related to the final `lm_head` output space?

This required revisiting what the logged training `mse` actually meant.

### What the Original Logged `mse` Actually Was

In the original training logs, each layer printed a field named `mse`.

However, this field was not always the same mathematical object across experiments:

- without `--svd_loss`, the logged `mse` was exactly the plain reconstruction MSE between the floating-point layer output and the quantized layer output
- with `--svd_loss`, the logged `mse` was actually the **SVD-weighted reconstruction loss**, because the same logging variable was reused for the optimized objective

Therefore, the old logs could not answer the question:

- whether the SVD-weighted method reduces error in the final output-head-related directions

This was a major source of ambiguity in the previous interpretation.

### Why the Logging Needed to Be Changed

Our mathematical motivation for SVD-weighted FlatQuant is not to reduce plain hidden-state MSE uniformly in every direction, but to reduce error along directions that matter more for the final `lm_head` output.

So, to diagnose the method correctly, we needed to log three different quantities separately:

1. the actual optimized objective
2. the plain hidden-state reconstruction MSE
3. an `lm_head`-related output-space proxy error

### First Failed Attempt and Why It Was Replaced

We first attempted to log an explicit head-output MSE by constructing the error after projection through the full output head:

- `logits_err = e @ W^T`

where:

- `e = y_fp - y_quant`
- `W = lm_head.weight`

This immediately caused CUDA OOM on both runs, because the vocabulary dimension is very large and the explicit logits error tensor is too expensive to materialize.

So this explicit version was discarded.

More explicitly, the quantity we originally intended to measure was the explicit output-head-space mean squared error:

- `head_mse_explicit = mean((e @ W^T)^2)`

with:

- `e = y_fp - y_quant`
- `W = lm_head.weight`

This is the most direct way to ask whether the quantization error becomes smaller after projection into the final logits space.

However, for Qwen2.5-3B the vocabulary dimension is very large, so for a batch of hidden states with shape roughly:

- `[bsz, seqlen, hidden] = [4, 2048, 2048]`

the explicit logits error would have shape roughly:

- `[4, 2048, vocab] = [4, 2048, 151936]`

This intermediate tensor is too large, so the explicit implementation caused immediate OOM on both diagnostic reruns. Therefore, the explicit head-space MSE could not be used in practice.

### Final Diagnostic Metrics Added to the Code

We then updated `flatquant/train_utils.py` so that each layer now logs:

- `optimized_loss`
- `plain_mse`
- `head_proxy_mse`

#### 1. `plain_mse`

This is the usual hidden-state reconstruction MSE:

- `plain_mse = mean((y_fp - y_quant)^2)`

#### 2. `optimized_loss`

This is the actual training objective used by the run:

- for baseline FlatQuant: `optimized_loss = plain_mse`
- for SVD-weighted FlatQuant: `optimized_loss = svd_weighted_loss`

So `optimized_loss` is not the same quantity across the two experiment types.

#### 3. `head_proxy_mse`

To avoid materializing the full logits-space error tensor, we used the quadratic form induced by the output head.

Let:

- `W = lm_head.weight = U diag(s) V^T`
- `e = y_fp - y_quant`

The explicit logits-space squared error is related to:

- `||eW^T||^2`

Using:

- `W^T W = V diag(s^2) V^T`

we define the proxy:

- `e_proj = eV`
- `head_proxy_mse = mean(e_proj^2 * s^2)`

This quantity is not the explicit per-logit MSE tensor itself, but it corresponds to the same `W^T W`-induced quadratic form and is therefore the correct low-memory proxy for the head-output-sensitive error we care about.

The mathematical reason this replacement is valid is the following:

- the explicit logits-space squared error is based on `||eW^T||^2`
- by expanding the quadratic form we get `e W^T W e^T`
- with `W = U diag(s) V^T`, we have `W^T W = V diag(s^2) V^T`

Therefore, the explicit head-space error and the proxy are governed by the same output-head-induced quadratic form. In other words:

- the explicit version asks for the squared error after projecting into the full logits space
- the proxy version evaluates the same geometry without explicitly materializing the logits tensor

So the replacement was not an ad-hoc engineering trick; it was a mathematically motivated low-memory reformulation of the same output-head-sensitive error structure.

### New Diagnostic Experiment Setup

To compare the two methods under the new diagnostics, we reran the two full configurations:

#### Baseline diagnostic run

- model: `Qwen2.5-3B` base
- config: `lwc + lac`
- GPU: `0`
- exp name: `qwen25_3b_base_w4a4kv4_lwc_lac_full_headmse_gpu0`

#### SVD diagnostic run

- model: `Qwen2.5-3B` base
- config: `lwc + lac + svd_loss`
- `svd_weight_mode = sigma2_norm`
- GPU: `3`
- exp name: `qwen25_3b_base_w4a4kv4_lwc_lac_svd_full_headmse_gpu3`

### Final PPL Results of the Diagnostic Re-runs

The diagnostic reruns reproduced the previous conclusion:

#### Baseline (`lwc + lac`)

- WikiText2 PPL: `8.8095`
- C4 PPL: `14.5661`

#### SVD-weighted (`lwc + lac + svd_loss`)

- WikiText2 PPL: `8.8400`
- C4 PPL: `14.5785`

So the final perplexity result remains unchanged:

- SVD-weighted FlatQuant is still slightly worse than the best stable baseline

### What the New Diagnostics Showed

The new logs revealed a much more nuanced picture than the old single `mse` field.

#### Observation 1. `optimized_loss` does not become systematically smaller

In many layers, especially later layers, the SVD run has a larger final `optimized_loss` than the baseline run.

Important note:

- this does **not** mean the SVD method necessarily failed mathematically, because `optimized_loss` is not the same function across the two runs

Still, it shows that the SVD-weighted objective is not easier to optimize in practice.

#### Observation 2. `plain_mse` is often slightly worse with SVD

Across many layers, the SVD run ends with a slightly larger plain hidden-state MSE than the non-SVD baseline.

This means the SVD method often sacrifices ordinary hidden-space reconstruction quality.

#### Observation 3. `head_proxy_mse` is often smaller with SVD, especially in important later layers

This is the most important new result.

For several layers, including some late layers near the output head, the SVD run obtains:

- larger `plain_mse`
- but smaller `head_proxy_mse`

This means the SVD-weighted method really is pushing the error toward directions that are less harmful under the `lm_head`-induced quadratic form.

### Representative Layer Comparisons

#### Layer 0

- baseline: `plain_mse = 0.17738493`, `head_proxy_mse = 16.23518562`
- SVD: `plain_mse = 0.17842296`, `head_proxy_mse = 15.79053974`

Interpretation:

- plain MSE is slightly worse
- head-sensitive proxy error is better

#### Layer 20

- baseline: `plain_mse = 0.38674530`, `head_proxy_mse = 37.50878906`
- SVD: `plain_mse = 0.39270559`, `head_proxy_mse = 37.00822067`

Interpretation:

- plain MSE is worse
- head-sensitive proxy error is better

#### Layer 34

- baseline: `plain_mse = 8.62847614`, `head_proxy_mse = 897.73602295`
- SVD: `plain_mse = 9.07591629`, `head_proxy_mse = 869.42364502`

Interpretation:

- plain MSE is clearly worse
- head-sensitive proxy error is clearly better

#### Layer 35

- baseline: `plain_mse = 9.59827709`, `head_proxy_mse = 1187.19238281`
- SVD: `plain_mse = 10.40928078`, `head_proxy_mse = 1125.40075684`

Interpretation:

- plain MSE is significantly worse
- head-sensitive proxy error is significantly better

This is especially important because layer 35 is the final transformer block, i.e. the closest hidden representation before the final norm and `lm_head`.

### Main Conclusion of This Diagnostic Round

The new diagnostics show that the current SVD-weighted method is **not** simply “worse in every respect”. Instead, the picture is:

- it often worsens ordinary hidden-state MSE
- it often does **not** look better in the raw optimized-loss scalar
- but it does improve the `lm_head`-related proxy error in several layers, especially some important late ones

So the most accurate interpretation is:

- the SVD weighting is doing something directionally meaningful
- it is indeed reshaping the reconstruction error toward directions that are more favorable under the output-head quadratic form
- but this improvement is currently not strong enough, or not globally coordinated enough, to produce better final perplexity

### Updated Research Interpretation

This means the current bottleneck is no longer best described as:

- “the SVD idea does nothing”

Instead, it is better described as:

- “the SVD idea does affect the error geometry in the intended direction, but the current loss design induces a trade-off that does not yet improve end-task performance”

### Implication for Next Steps

Future SVD work should focus on keeping the gain in `head_proxy_mse` while reducing the damage to `plain_mse`. The most natural next directions are:

1. use a weaker spectral weighting than `sigma^2`
2. switch from full replacement to a mixed objective
3. keep the new diagnostic logs permanently, because a single `mse` field is not enough to interpret SVD-vs-baseline behavior correctly

## 2026-04-13

### Goal Update

After completing the PPL-based comparison and the head-proxy diagnostic round, we decided to extend the downstream comparison with additional zero-shot tasks instead of continuing to focus on `WikiText2` / `C4`.

This decision was also motivated by the fact that the explicit head-output MSE path had already been judged too memory-expensive to keep using in routine experiments. The low-memory `head_proxy_mse` diagnostic remains sufficient for internal analysis, while the next empirical question is whether SVD-weighted FlatQuant helps on a broader set of zero-shot tasks.

### Engineering Change for This Round

To support a zero-shot-only evaluation run, we added a CLI switch:

- `--skip_ppl_eval`

This allows `main.py` to skip the fixed `WikiText2` / `C4` perplexity loop and directly run the requested LM Eval tasks.

### Zero-Shot Evaluation Setup

Both runs used:

- model: `Qwen2.5-3B` base
- quantization: `W4A4KV4`
- `nsamples=128`
- `epochs=15`
- `cali_bsz=4`
- `flat_lr=5e-3`
- `--cali_trans --add_diag --lwc --lac --deactive_amp --direct_inv`
- GPU: `0`

The compared runs were:

#### Baseline zero-shot run

- exp name: `qwen25_3b_base_w4a4kv4_lwc_lac_zeroshot_gpu0`
- no `--svd_loss`

#### SVD zero-shot run

- exp name: `qwen25_3b_base_w4a4kv4_lwc_lac_svd_zeroshot_gpu0`
- `--svd_loss`
- `--svd_file /gammadisk/liuxuanang/proj/SVD_A/results/svd/_gammadisk_liuxuanang_proj_FlatQuant_modelzoo_Qwen_Qwen2.5-3B_svd.npz`
- `--svd_weight_mode sigma2_norm`

Requested zero-shot task list:

- `piqa`
- `hellaswag`
- `arc_easy`
- `arc_challenge`
- `winogrande`
- `lambada_openai`
- `boolq`
- `openbookqa`
- `social_iqa`

### What Actually Completed

Both runs successfully completed the first six tasks and produced valid scores for:

- `piqa`
- `hellaswag`
- `arc_easy`
- `arc_challenge`
- `winogrande`
- `lambada_openai`

However, both runs stopped during `boolq` evaluation and therefore did **not** produce results for:

- `boolq`
- `openbookqa`
- `social_iqa`

So this round should be interpreted as a **partial but still useful zero-shot comparison**, not a full 9-task completion.

### Completed Zero-Shot Results

| Task | Baseline | SVD | Delta (SVD - Baseline) |
| --- | ---: | ---: | ---: |
| `piqa` | `76.22` | `75.95` | `-0.27` |
| `hellaswag` | `70.50` | `70.50` | `0.00` |
| `arc_easy` | `73.27` | `72.05` | `-1.22` |
| `arc_challenge` | `45.99` | `44.37` | `-1.62` |
| `winogrande` | `65.75` | `65.19` | `-0.56` |
| `lambada_openai` | `63.21` | `63.19` | `-0.02` |

Average over the six completed tasks:

- baseline 6-task avg: `65.82`
- SVD 6-task avg: `65.21`
- delta: `-0.61`

### Interpretation of This Round

Under the currently completed zero-shot subset:

- SVD did **not** outperform the non-SVD FlatQuant baseline
- `hellaswag` was effectively tied
- all other completed tasks were slightly worse with SVD

So this round is directionally consistent with the earlier PPL conclusion:

- the current SVD-weighted objective changes the error geometry in a meaningful way
- but this still does not translate into better downstream accuracy under the present loss design

### Practical Conclusion

At this point, the evidence from three views is aligned:

1. PPL comparison: SVD is slightly worse
2. head-proxy diagnostics: SVD improves some output-head-sensitive geometry but often worsens plain MSE
3. partial zero-shot comparison: SVD is again slightly worse overall on the completed tasks

Therefore, the current conclusion remains:

- the present `sigma2_norm` full-replacement SVD loss is analytically interesting and functionally stable
- but it still does **not** beat the stable non-SVD FlatQuant baseline on `Qwen2.5-3B` base

### Follow-Up Note

Because both runs stopped during `boolq`, the remaining three tasks were not used for the official comparison in this round. If needed later, they should be rerun separately as task-isolated evaluations rather than by reusing the interrupted 9-task batch as if it were complete.

## 2026-04-15: `sigma2_norm_clip_low` SVD Variant

Following the suggestion to reduce damage on small-singular-value directions, we added a new SVD weighting mode that only keeps SVD amplification on directions whose singular values are larger than the mean singular value.

### Code Change

We added a new CLI mode:

- `--svd_weight_mode sigma2_norm_clip_low`

Affected code paths for this update:

- `flatquant/args_utils.py`
- `flatquant/train_utils.py`

Its implementation is:

1. first compute the normalized SVD weights
   - `w = s^2 / mean(s^2)`
2. then clip low-singular-value directions back to unit weight
   - if `s_i > mean(s)`, keep `w_i`
   - if `s_i <= mean(s)`, set `w_i = 1`

This keeps ordinary MSE weighting on low-singular-value directions while preserving stronger weighting on high-singular-value directions.

The corresponding scripts for this update are:

- `scripts/qwen-2.5-base/qwen-2.5-3b/w4a4kv4_svd_clip_low_full_eval.sh`
- `scripts/qwen-2.5-base/qwen-2.5-3b/w4a4kv4_svd_clip_low_zeroshot.sh`
- `scripts/qwen-2.5-base/qwen-2.5-3b/w4a4kv4_svd_clip_low_full_eval_auto.sh`

### Formal Run Configuration

The formal run used the same main FlatQuant setup as the established baseline, with the only SVD-specific difference being the new weight mode:

- model: `Qwen2.5-3B base`
- quantization: `W4A4KV4`
- calibration/training flags:
  - `--cali_trans --add_diag --lwc --lac --deactive_amp --direct_inv`
- SVD flags:
  - `--svd_loss`
  - `--svd_file /gammadisk/liuxuanang/proj/SVD_A/results/svd/_gammadisk_liuxuanang_proj_FlatQuant_modelzoo_Qwen_Qwen2.5-3B_svd.npz`
  - `--svd_weight_mode sigma2_norm_clip_low`
- evaluation scope:
  - `WikiText2`
  - `C4`
  - six zero-shot tasks:
    - `piqa`
    - `hellaswag`
    - `arc_easy`
    - `arc_challenge`
    - `winogrande`
    - `lambada_openai`

### Formal Run Results

Formal run log directory:

- `outputs/Qwen2.5-3B/w4a4/qwen25_3b_base_w4a4kv4_lwc_lac_svd_clip_low_full_eval_gpu3/`

#### PPL

- `WikiText2`: `8.81881332397461`
- `C4`: `14.60970401763916`

#### Six-task zero-shot

- `piqa`: `77.15`
- `hellaswag`: `70.55`
- `arc_easy`: `71.00`
- `arc_challenge`: `45.48`
- `winogrande`: `65.67`
- `lambada_openai`: `63.54`
- `6-task avg`: `65.57`

#### Head-proxy diagnostic snapshots from the formal run

The training log continues to print `optimized_loss`, `plain_mse`, and `head_proxy_mse` at every layer and epoch. The final-epoch (`iter 14`) `head_proxy_mse` values for representative layers in the formal run were:

- layer 10: `36.75202942`
- layer 20: `36.26086807`
- layer 34: `865.69470215`
- layer 35: `1117.51342773`

### Appendix: Non-formal Run With Runtime Issues

Before the final formal run above, we also executed an earlier `sigma2_norm_clip_low` run on GPU3 that completed evaluation but is not treated as the formal result for this round.

Appendix run log directory:

- earlier GPU3 run under the same experiment name prior to the final rerun

#### Problem observed in that run

During training, the log contained repeated runtime errors from the MAGMA/CUBLAS linear algebra path, including:

- `magma_dgetrs_gpu`
- `CUBLAS error: out of memory`
- `memory mapping error`

These messages appeared during the layerwise training stage.

#### Appendix run data

- `WikiText2`: `8.816495895385742`
- `C4`: `14.604449272155762`
- `piqa`: `76.28`
- `hellaswag`: `70.42`
- `arc_easy`: `70.08`
- `arc_challenge`: `45.65`
- `winogrande`: `65.82`
- `lambada_openai`: `63.46`
- `6-task avg`: `65.28`

## 2026-04-15: A+B Full-Scale Run Plan

### Goal

For the next SVD step, this round is restricted to the combined direction:

- A: mixed loss
- B: layer-dependent SVD strength schedule

No reduced-budget screening is used for this round. The run is launched directly under the fixed normal-scale protocol.

### Method Definition

The intended training objective is layer-dependent mixed loss:

- `loss_i = (1 - alpha_i) * plain_mse + alpha_i * svd_weighted_loss`

with linear layer schedule:

- `alpha_i` increases linearly from `0.0` at the first transformer layer to `0.5` at the last transformer layer

SVD weighting configuration for this run:

- `svd_weight_mode = sigma2_norm`

### Formal Run Configuration

- model: `Qwen2.5-3B` base
- quantization: `W4A4KV4`
- calibration/training flags:
  - `--cali_trans --add_diag --lwc --lac --deactive_amp --direct_inv`
- normal-scale budget:
  - `nsamples=128`
  - `epochs=15`
  - `cali_bsz=4`
  - `flat_lr=5e-3`
- SVD / A+B flags:
  - `--svd_loss`
  - `--svd_file /gammadisk/liuxuanang/proj/SVD_A/results/svd/_gammadisk_liuxuanang_proj_FlatQuant_modelzoo_Qwen_Qwen2.5-3B_svd.npz`
  - `--svd_weight_mode sigma2_norm`
  - `--svd_loss_alpha 0.5`
  - `--svd_alpha_schedule linear`
  - `--svd_alpha_start 0.0`
- evaluation scope:
  - `WikiText2`
  - `C4`
  - six zero-shot tasks:
    - `piqa`
    - `hellaswag`
    - `arc_easy`
    - `arc_challenge`
    - `winogrande`
    - `lambada_openai`
- target GPU:
  - GPU `3`

Corresponding scripts:

- `scripts/qwen-2.5-base/qwen-2.5-3b/w4a4kv4_svd_mix_linear_full_eval.sh`
- `scripts/qwen-2.5-base/qwen-2.5-3b/w4a4kv4_svd_mix_linear_full_eval_gpu3.sh`

### Logging Rule For This Run

As with the current diagnostic-enabled codepath, training logs for this run should continue to distinguish:

- `optimized_loss`
- `plain_mse`
- `head_proxy_mse`

### Code Change

To support the A+B method, we extended the current SVD-loss path so that the optimized objective is no longer limited to two extremes (`plain_mse` only or full SVD replacement only).

Affected code paths for this update:

- `flatquant/args_utils.py`
- `flatquant/train_utils.py`
- `scripts/qwen-2.5-base/qwen-2.5-3b/w4a4kv4_svd_mix_linear_full_eval.sh`
- `scripts/qwen-2.5-base/qwen-2.5-3b/w4a4kv4_svd_mix_linear_full_eval_gpu3.sh`

New CLI arguments added for this round:

- `--svd_loss_alpha`
- `--svd_alpha_schedule`
- `--svd_alpha_start`

Current semantics:

- `--svd_loss_alpha`
  - final-layer SVD mix coefficient
- `--svd_alpha_schedule`
  - how to vary the SVD mix across layers
- `--svd_alpha_start`
  - first-layer SVD mix coefficient when using a scheduled mix

For this round, the implemented schedule choices are:

- `constant`
- `linear`

The current code also logs one additional low-cost diagnostic value when SVD loss is enabled:

- SVD weight dynamic range

which reports:

- minimum SVD weight
- maximum SVD weight
- `max / min` ratio

### Mathematical Rationale

The motivation for A+B is to preserve the useful directional bias introduced by the output-head spectrum while preventing the full loss-replacement design from over-sacrificing ordinary hidden-state reconstruction.

Let:

- `e = y_fp - y_quant`
- `W = lm_head.weight = U diag(s) V^T`
- `e_proj = eV`

Then:

- `plain_mse = mean(e^2)`
- `svd_weighted_loss = mean(e_proj^2 * w)`

where this run uses:

- `w = s^2 / mean(s^2)`

The A+B mixed objective is:

- `loss_i = (1 - alpha_i) * plain_mse + alpha_i * svd_weighted_loss`

with layer-dependent linear schedule:

- `alpha_i = alpha_start + (alpha_end - alpha_start) * i / (L - 1)`

For the formal run in this round:

- `alpha_start = 0.0`
- `alpha_end = 0.5`

So the first transformer layer remains pure `plain_mse`, while later layers gradually receive a stronger output-head-sensitive directional bias.

This design was chosen because the previous diagnostic round suggested two things simultaneously:

1. full SVD replacement can improve `head_proxy_mse` in important later layers
2. full replacement also tends to worsen `plain_mse`, which can damage the layerwise reconstruction chain

Therefore, the mixed objective attempts to retain the later-layer directional benefit while reducing damage to ordinary reconstruction quality, especially in earlier layers.

### Formal Run Log Directory

- `outputs/Qwen2.5-3B/w4a4/qwen25_3b_base_w4a4kv4_lwc_lac_svd_mix_linear_a0p5_full_eval_gpu3_20260415_180145/`

Launcher log:

- `outputs/qwen25_3b_base_w4a4kv4_lwc_lac_svd_mix_linear_a0p5_full_eval_gpu3_20260415_180145.launcher.log`

### Formal Run Results

#### PPL

- `WikiText2`: `8.817450523376465`
- `C4`: `14.568132400512695`

#### Six-task zero-shot

- `piqa`: `77.09`
- `hellaswag`: `70.72`
- `arc_easy`: `71.55`
- `arc_challenge`: `45.48`
- `winogrande`: `67.09`
- `lambada_openai`: `63.75`
- `6-task avg`: `65.95`

#### Representative final-layer diagnostic snapshot

Final epoch (`iter 14`) of layer 35 from the formal run:

- `optimized_loss`: `11.14162159`
- `plain_mse`: `9.95463371`
- `head_proxy_mse`: `1114.93212891`

### Comparison Against The Established Full Baseline

Reference baseline from the existing formal normal-scale run:

- `WikiText2`: `8.8095`
- `C4`: `14.5661`
- `6-task avg`: `65.82`

Current A+B formal run:

- `WikiText2`: `8.817450523376465`
- `C4`: `14.568132400512695`
- `6-task avg`: `65.95`

### Interpretation Of This Round

Under the fixed normal-scale protocol, the A+B method produced the following pattern:

- `WikiText2` is slightly worse than the established baseline
- `C4` is also slightly worse than the established baseline
- the six-task zero-shot average is slightly better than the established baseline

So the current empirical picture is more favorable than the earlier full-replacement `sigma2_norm` SVD attempt, but it still does not provide a clean across-the-board win over the established baseline.

More precisely:

- on perplexity, A+B is extremely close to baseline but still slightly behind
- on the six completed zero-shot tasks, A+B is slightly ahead on the average

Therefore, the most accurate current conclusion is:

- A+B is a meaningful improvement over the earlier full-replacement SVD design
- it narrows the PPL gap substantially and slightly improves the six-task zero-shot average
- but it still has not yet produced a full dominant result over the established normal-scale baseline on `Qwen2.5-3B` base

## 2026-05-28: Qwen2.5-3B Base Outlier Distribution Diagnostic

### Goal

Following the distribution-analysis suggestion, this round checked whether the current SVD / A+B method changes the quantization-relevant outlier structure, rather than only checking final PPL or zero-shot scores.

The diagnostic question was:

- after the transformation, do weight outliers and token / activation outliers become more quantization-friendly?
- does A+B SVD-FlatQuant add a clear distribution-flattening effect beyond the established FlatQuant baseline?

### Teacher-Suggestion Requirements Reflected In This Round

The useful requirements extracted from the teacher comments were:

- compare before and after transformation
- use concrete metrics, not only final task scores
- use multi-scale plots / histograms / heatmaps
- include signal-to-noise style metrics
- inspect whether adding SVD actually changes the weight / activation distribution
- connect the distribution view with the SVD goal of preserving more semantically important directions

This round implements the metric / plot / SQNR parts. KL-style histogram divergence was not included in this run and is recorded as a follow-up item.

### Code Change

Added a diagnostic script:

- `diagnostics/qwen25_outlier_distribution.py`

The script loads existing `flat_parameters.pth` files and does not rerun full calibration.

### Compared States

The diagnostic compared:

| State | Meaning | Source |
| --- | --- | --- |
| `fp` | original floating-point Qwen2.5-3B base | `./modelzoo/Qwen/Qwen2.5-3B` |
| `baseline` | established FlatQuant baseline | `outputs/Qwen2.5-3B/w4a4/qwen25_3b_base_w4a4kv4_lwc_lac_full_headmse_gpu0/` |
| `svd_mix_linear` | A+B SVD-FlatQuant formal run | `outputs/Qwen2.5-3B/w4a4/qwen25_3b_base_w4a4kv4_lwc_lac_svd_mix_linear_a0p5_full_eval_gpu3_20260415_180145/` |

### Run Command

The run used GPU `0` and the `flatquant_svd` conda environment:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n flatquant_svd python diagnostics/qwen25_outlier_distribution.py --nsamples 4
```

### Output Directory

- `outputs/diagnostics/qwen25_3b_base_outlier_dist_baseline_vs_svd_mix_linear_20260528_005938/`

Main outputs:

- `summary.json`
- `weight_stats_all.csv`
- `activation_stats_all.csv`
- per-state CSV files
- heatmaps
- representative histograms

Full report:

- `docs/report_outlier_distribution_qwen25_base_20260528.md`

### Metrics

Weight metrics included:

- `abs_max_over_p99`
- `p999_over_p50`
- `per_out_max_over_median`
- `w4_sym_per_out_mse`
- `w4_sym_per_out_sqnr_db`

Activation / token metrics included:

- `abs_max_over_p99`
- `p999_over_p50`
- `token_score_p99`
- `channel_max_over_median`
- `a4_sym_token_mse`
- `a4_sym_token_sqnr_db`

where token outlier score is:

- `max(abs(token)) / mean(abs(token))`

### Global Weight Summary

| Metric | FP | Baseline | SVD A+B |
| --- | ---: | ---: | ---: |
| `abs_max_over_p99` mean | `6.6690` | `2.6596` | `2.5746` |
| `p999_over_p50` mean | `10.0473` | `5.6902` | `5.6919` |
| `per_out_max_over_median` mean | `5.0389` | `3.0766` | `2.9404` |
| `w4_sym_per_out_mse` mean | `2.2760e-05` | `9.6427e-05` | `9.7644e-05` |
| `w4_sym_per_out_sqnr_db` mean | `14.9394` | `19.4555` | `19.4392` |

### Global Activation Summary

| Metric | FP | Baseline | SVD A+B |
| --- | ---: | ---: | ---: |
| `abs_max_over_p99` mean | `218.4837` | `2.9569` | `2.9881` |
| `p999_over_p50` mean | `58.5018` | `5.8683` | `5.8460` |
| `token_score_p99` mean | `119.0528` | `5.8646` | `5.8688` |
| `channel_max_over_median` mean | `6511.8537` | `1.6619` | `1.6744` |
| `a4_sym_token_mse` mean | `0.155963` | `0.000833` | `0.000820` |
| `a4_sym_token_sqnr_db` mean | `6.4133` | `16.3468` | `16.3404` |

### Baseline vs SVD A+B Improvement Counts

Across all `252` layer-linear modules:

| Metric | SVD A+B Improved / Total | Mean Delta `(SVD - Baseline)` | Median Delta |
| --- | ---: | ---: | ---: |
| weight `abs_max_over_p99` | `146 / 252` | `-0.08501374` | `-0.01417280` |
| weight `per_out_max_over_median` | `146 / 252` | `-0.13618744` | `-0.01849365` |
| weight `w4_sym_per_out_sqnr_db` | `66 / 252` | `-0.01631684` | `-0.01040521` |
| activation `token_score_p99` | `106 / 252` | `0.00415793` | `0.00479412` |
| activation `channel_max_over_median` | `133 / 252` | `0.01247359` | `-0.01031278` |
| activation `a4_sym_token_sqnr_db` | `94 / 252` | `-0.00639549` | `-0.00396397` |
| activation `a4_sym_token_mse` | `127 / 252` | `-0.00001257` | approximately `0` |

### Representative Late-Layer Data

Layer 34 `o_proj` activation:

| Metric | FP | Baseline | SVD A+B |
| --- | ---: | ---: | ---: |
| `token_score_p99` | `24.683493` | `7.686549` | `7.694026` |
| `channel_max_over_median` | `2.573427` | `2.275362` | `2.062016` |
| `a4_sym_token_sqnr_db` | `9.562100` | `15.444544` | `15.357902` |
| `a4_sym_token_mse` | `0.069418` | `0.012709` | `0.011612` |

Layer 35 `down_proj` activation:

| Metric | FP | Baseline | SVD A+B |
| --- | ---: | ---: | ---: |
| `token_score_p99` | `470.694580` | `6.062948` | `6.152538` |
| `channel_max_over_median` | `132.961240` | `1.952569` | `1.966387` |
| `a4_sym_token_sqnr_db` | `8.638284` | `15.602676` | `15.575827` |
| `a4_sym_token_mse` | `0.411832` | `0.000892` | `0.000808` |

Layer 35 `down_proj` weight:

| Metric | FP | Baseline | SVD A+B |
| --- | ---: | ---: | ---: |
| `abs_max_over_p99` | `21.372263` | `4.275304` | `3.801527` |
| `per_out_max_over_median` | `13.130045` | `4.327869` | `3.860465` |
| `w4_sym_per_out_sqnr_db` | `13.073733` | `19.885132` | `19.825805` |
| `w4_sym_per_out_mse` | `0.000030` | `0.000104` | `0.000116` |

### Interpretation

The diagnostic shows that the established FlatQuant baseline has a strong distribution effect:

- FP activation `token_score_p99` mean: `119.0528`
- baseline activation `token_score_p99` mean: `5.8646`
- FP activation `a4_sym_token_sqnr_db` mean: `6.4133`
- baseline activation `a4_sym_token_sqnr_db` mean: `16.3468`

So baseline FlatQuant already changes the FP activation/token distributions into a much more quantization-friendly form.

The current A+B SVD method does not add a clear global distribution-flattening effect over that baseline:

- baseline activation `token_score_p99` mean: `5.8646`
- SVD A+B activation `token_score_p99` mean: `5.8688`
- baseline activation `a4_sym_token_sqnr_db` mean: `16.3468`
- SVD A+B activation `a4_sym_token_sqnr_db` mean: `16.3404`

For weights, SVD A+B slightly reduces some outlier ratios, but does not improve W4 SQNR on average:

- baseline weight `abs_max_over_p99` mean: `2.6596`
- SVD A+B weight `abs_max_over_p99` mean: `2.5746`
- baseline weight `w4_sym_per_out_sqnr_db` mean: `19.4555`
- SVD A+B weight `w4_sym_per_out_sqnr_db` mean: `19.4392`

### Mechanism-Level Conclusion

This round supports the following interpretation:

- the established FlatQuant baseline already performs the major outlier-removal / distribution-flattening work
- current A+B SVD-FlatQuant does not clearly make the ordinary weight / activation distributions more quantization-friendly than baseline
- A+B SVD is better understood as an output-head-sensitive error-geometry reweighting method, consistent with the previous `head_proxy_mse` diagnostic
- the lack of a strong additional distribution improvement is a plausible reason why A+B does not produce a clean across-the-board PPL win

### Follow-Up

Recommended follow-up diagnostics:

1. Add KL / JS divergence on histograms for FP-vs-baseline and baseline-vs-SVD distribution comparison.
2. Jointly plot `plain_mse`, `head_proxy_mse`, token outlier metrics, and A4/W4 SQNR for the same layers.
3. Use the joint view to test whether layers with improved `head_proxy_mse` also show any distribution-side improvement, or whether the benefit is purely output-head-geometry based.

## 2026-05-29: Qwen2.5-3B Base KL/JS And Head-Proxy Alignment Diagnostic

### Goal

This round followed up on the 2026-05-28 outlier-distribution diagnostic by adding two analyses requested by the teacher-style feedback:

- KL / JS-style histogram divergence to quantify distribution changes
- joint alignment between `head_proxy_mse` improvement and ordinary outlier / SQNR metrics

The main question was:

- does A+B SVD-FlatQuant actually change the ordinary weight / activation distributions beyond baseline?
- when `head_proxy_mse` improves, do token outliers and A4/W4 SQNR improve at the same time?

### Code Change

Added:

- `diagnostics/qwen25_kl_js_head_alignment.py`

### Run Command

The run used GPU `0` and the `flatquant_svd` conda environment:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n flatquant_svd python diagnostics/qwen25_kl_js_head_alignment.py --nsamples 4
```

### Output Directory

- `outputs/diagnostics/qwen25_3b_base_kl_js_head_alignment_20260529_234646/`

Main outputs:

- `summary.json`
- `kl_js_distribution_rows.csv`
- `head_outlier_alignment_rows.csv`
- JS heatmaps
- head-proxy-vs-token-outlier scatter plot

Reports:

- `docs/report_kl_js_head_alignment_qwen25_base_20260529_zh.md`
- `docs/report_explain_outlier_and_kljs_diagnostics_zh.md`

### JS Divergence Results

The diagnostic mainly reports JS divergence because it is symmetric and more stable than raw KL for histogram comparison.

#### Activation absolute-value distribution

| Pair | Mean JS | Median JS | Max JS | Count |
| --- | ---: | ---: | ---: | ---: |
| FP vs Baseline | `0.132993` | `0.139836` | `0.270863` | `32` |
| Baseline vs SVD A+B | `0.000223` | `0.000131` | `0.000967` | `32` |
| FP vs SVD A+B | `0.134866` | `0.136159` | `0.276501` | `32` |

#### Token outlier-score distribution

| Pair | Mean JS | Median JS | Max JS | Count |
| --- | ---: | ---: | ---: | ---: |
| FP vs Baseline | `0.687812` | `0.693089` | `0.693147` | `32` |
| Baseline vs SVD A+B | `0.000444` | `0.000214` | `0.002077` | `32` |
| FP vs SVD A+B | `0.687720` | `0.693147` | `0.693147` | `32` |

#### Weight absolute-value distribution

| Pair | Mean JS | Median JS | Max JS | Count |
| --- | ---: | ---: | ---: | ---: |
| FP vs Baseline | `0.136758` | `0.117388` | `0.297434` | `32` |
| Baseline vs SVD A+B | `0.000585` | `0.000201` | `0.002869` | `32` |
| FP vs SVD A+B | `0.137609` | `0.116241` | `0.307274` | `32` |

### JS Interpretation

The JS results show:

- FP-to-baseline distribution change is large
- FP-to-SVD-A+B distribution change is also large
- baseline-to-SVD-A+B distribution change is extremely small

For token outlier scores, the contrast is especially clear:

- FP vs baseline mean JS: `0.687812`
- baseline vs SVD A+B mean JS: `0.000444`

This confirms that the ordinary distribution transformation is mainly caused by the FlatQuant baseline, not by the additional SVD A+B objective.

### Head-Proxy / Outlier Alignment Results

The alignment part compared final-epoch training metrics from the baseline and A+B logs with ordinary distribution metrics from the outlier diagnostic.

Across all `252` layer-linear modules:

| Metric | Improved Modules |
| --- | ---: |
| `head_proxy_mse` improved | `231 / 252` |
| `plain_mse` improved | `70 / 252` |
| activation `token_score_p99` improved | `106 / 252` |
| activation `a4_sym_token_sqnr_db` improved | `94 / 252` |
| weight `abs_max_over_p99` improved | `146 / 252` |
| weight `w4_sym_per_out_sqnr_db` improved | `66 / 252` |

This shows that A+B SVD improves the head-sensitive proxy in most layers, but that improvement does not synchronize with ordinary hidden-space MSE, token outlier metrics, or A4/W4 SQNR.

### Representative Late-Layer Alignment

#### Layer 34

| Linear | Head Proxy Rel Delta | Plain MSE Rel Delta | Token Score Delta | A4 SQNR Delta |
| --- | ---: | ---: | ---: | ---: |
| `q_proj` | `-0.0405` | `0.0252` | `-0.0291` | `-0.0006` |
| `o_proj` | `-0.0405` | `0.0252` | `0.0075` | `-0.0866` |
| `up_proj` | `-0.0405` | `0.0252` | `0.0417` | `-0.0388` |
| `down_proj` | `-0.0405` | `0.0252` | `0.1118` | `-0.0437` |

#### Layer 35

| Linear | Head Proxy Rel Delta | Plain MSE Rel Delta | Token Score Delta | A4 SQNR Delta |
| --- | ---: | ---: | ---: | ---: |
| `q_proj` | `-0.0609` | `0.0371` | `-0.0041` | `-0.0335` |
| `o_proj` | `-0.0609` | `0.0371` | `0.1381` | `-0.0535` |
| `up_proj` | `-0.0609` | `0.0371` | `-0.0492` | `0.0325` |
| `down_proj` | `-0.0609` | `0.0371` | `0.0896` | `-0.0268` |

In late layers, A+B improves `head_proxy_mse` clearly, but `plain_mse` becomes worse and activation SQNR usually does not improve.

### Mechanism-Level Conclusion

This round confirms the interpretation from the previous distribution diagnostic:

- FlatQuant baseline is the main outlier-removal / distribution-flattening mechanism.
- A+B SVD mainly changes output-head-sensitive error geometry.
- The `head_proxy_mse` improvement does not automatically become ordinary activation/weight distribution improvement.
- This explains why A+B can look meaningful under `head_proxy_mse`, while still not producing a clean across-the-board PPL or zero-shot win.

### Follow-Up Direction

If the SVD line continues, the next method should not only adjust SVD alpha or spectral weights. It should explicitly preserve the `head_proxy_mse` gain while preventing damage to:

- `plain_mse`
- activation token outliers
- A4/W4 SQNR

One possible direction is to add an activation quantization-noise or SQNR proxy term to the mixed objective.
