#!/bin/bash

set -u

ROOT_DIR="/gammadisk/liuxuanang/proj/FlatQuant"
MODEL_PATH="./modelzoo/Qwen/Qwen2.5-3B"
SVD_FILE="/gammadisk/liuxuanang/proj/SVD_A/results/svd/_gammadisk_liuxuanang_proj_FlatQuant_modelzoo_Qwen_Qwen2.5-3B_svd.npz"
OUTPUT_DIR="./outputs"
EXP_PREFIX="qwen25_3b_base_w4a4kv4_lwc_lac_svd_clip_low_full_eval"
WATCH_LOG="${ROOT_DIR}/outputs/${EXP_PREFIX}.watcher.log"

mkdir -p "${ROOT_DIR}/outputs"

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    gpu_line=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '($1 >= 0 && $1 <= 3) && ($2 < 1000) {print $0; exit}')

    if [ -n "${gpu_line}" ]; then
        gpu_id=$(printf '%s' "${gpu_line}" | cut -d',' -f1 | tr -d ' ')
        exp_name="${EXP_PREFIX}_gpu${gpu_id}"
        exp_dir="${ROOT_DIR}/outputs/Qwen2.5-3B/w4a4/${exp_name}"
        mkdir -p "${exp_dir}"

        printf '[%s] Found idle GPU %s with status: %s\n' "${timestamp}" "${gpu_id}" "${gpu_line}" | tee -a "${WATCH_LOG}"

        nohup bash -lc "cd \"${ROOT_DIR}\" && CUDA_VISIBLE_DEVICES=${gpu_id} python ./main.py --model ${MODEL_PATH} --w_bits 4 --a_bits 4 --k_bits 4 --k_asym --k_groupsize 128 --v_bits 4 --v_asym --v_groupsize 128 --nsamples 128 --epochs 15 --cali_bsz 4 --flat_lr 5e-3 --cali_trans --add_diag --lwc --lac --deactive_amp --direct_inv --svd_loss --svd_file ${SVD_FILE} --svd_weight_mode sigma2_norm_clip_low --output_dir ${OUTPUT_DIR} --exp_name ${exp_name} --lm_eval --lm_eval_batch_size 16 --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada_openai" > "${exp_dir}/nohup.log" 2>&1 &
        run_pid=$!

        printf '[%s] Started experiment on GPU %s with launcher PID %s\n' "${timestamp}" "${gpu_id}" "${run_pid}" | tee -a "${WATCH_LOG}"
        printf '[%s] Log path: %s/nohup.log\n' "${timestamp}" "${exp_dir}" | tee -a "${WATCH_LOG}"
        exit 0
    fi

    printf '[%s] No idle GPU among 0-3, retrying in 60s\n' "${timestamp}" | tee -a "${WATCH_LOG}"
    sleep 60
done
