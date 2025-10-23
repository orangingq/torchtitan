#!/usr/bin/bash

# Define common environment variables
EXPLAIN="Main Table Experiment"
EXPERIMENT_TAG="1021_4gpus"
TODAY="1021"

export WANDB_TAG="${EXPERIMENT_TAG}"
export CUDA_VISIBLE_DEVICES=1

LOG_DIR="/opt/dlami/nvme/DMLAB/shcho/torchtitan/logs/h200/${WANDB_TAG}"
CONFIG_FILE="${LOG_DIR}/config.toml"

for PP_SCHEDULER in gpipe 1f1b interleaved1f1b ; do # 1f1b gpipe interleaved1f1b  interleavedzb zbv 
    for METRIC_TYPE in fullrand7 nofreeze ; do 
        OUTPUT_FILE="${LOG_DIR}/${TODAY}_${PP_SCHEDULER}_${METRIC_TYPE}.log"
        BASENAME="${TODAY}_${PP_SCHEDULER}_${METRIC_TYPE}_dm4"

CUDA_VISIBLE_DEVICES=0,1,2 python3 -m timelyfreeze.eval_hf_checkpoint --model_path=/data2/shcho/torchtitan/checkpoint/1020_gpipe_nofreeze_dm4/step-500/sharded --dtype=float16 --output_json=/data2/shcho/torchtitan/checkpoint/1020_gpipe_nofreeze_dm4/step-500/eval_results.json


| tee -a ${OUTPUT_FILE}