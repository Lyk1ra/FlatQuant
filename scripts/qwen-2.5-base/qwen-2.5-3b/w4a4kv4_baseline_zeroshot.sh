#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./main.py \
    --model ./modelzoo/Qwen/Qwen2.5-3B \
    --w_bits 4 --a_bits 4 \
    --k_bits 4 --k_asym --k_groupsize 128 \
    --v_bits 4 --v_asym --v_groupsize 128 \
    --nsamples 128 --epochs 15 --cali_bsz 4 --flat_lr 5e-3 \
    --cali_trans --add_diag --lwc --lac --deactive_amp --direct_inv \
    --output_dir ./outputs \
    --exp_name qwen25_3b_base_w4a4kv4_lwc_lac_zeroshot_gpu0 \
    --skip_ppl_eval \
    --lm_eval --lm_eval_batch_size 16 \
    --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada_openai boolq openbookqa social_iqa
