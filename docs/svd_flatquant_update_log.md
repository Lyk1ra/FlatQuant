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
