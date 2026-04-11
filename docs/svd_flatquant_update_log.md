# SVD-Weighted FlatQuant Update Log

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
