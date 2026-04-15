#!/bin/bash

timestamp=$(date '+%Y%m%d_%H%M%S')

CUDA_VISIBLE_DEVICES=3 python ./main.py \
    --model ./modelzoo/Qwen/Qwen2.5-3B \
    --w_bits 4 --a_bits 4 \
    --k_bits 4 --k_asym --k_groupsize 128 \
    --v_bits 4 --v_asym --v_groupsize 128 \
    --nsamples 128 --epochs 15 --cali_bsz 4 --flat_lr 5e-3 \
    --cali_trans --add_diag --lwc --lac --deactive_amp --direct_inv \
    --svd_loss \
    --svd_file /gammadisk/liuxuanang/proj/SVD_A/results/svd/_gammadisk_liuxuanang_proj_FlatQuant_modelzoo_Qwen_Qwen2.5-3B_svd.npz \
    --svd_weight_mode sigma2_norm \
    --svd_loss_alpha 0.5 \
    --svd_alpha_schedule linear \
    --svd_alpha_start 0.0 \
    --output_dir ./outputs \
    --exp_name qwen25_3b_base_w4a4kv4_lwc_lac_svd_mix_linear_a0p5_full_eval_gpu3_${timestamp} \
    --lm_eval --lm_eval_batch_size 16 \
    --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada_openai
