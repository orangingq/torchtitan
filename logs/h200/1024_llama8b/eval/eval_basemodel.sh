#!/usr/bin/bash
#SBATCH --job-name=eval_basemodel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/h200/1024_llama8b/eval/slurm-%j.out

# Define common environment variables
EXPLAIN="Llama 3.1 8B Instruct Base Model Evaluation"
TODAY="1024"

# Respect Slurm's CUDA_VISIBLE_DEVICES
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="${1:-0}"
    echo "✔️ Using manually set GPU(s): ${CUDA_VISIBLE_DEVICES}"
else
    echo "✔️ SLURM JOB GPUS: ${SLURM_JOB_GPUS}"
    echo "✔️ Using Slurm-assigned GPU(s): ${CUDA_VISIBLE_DEVICES}"
fi

THIS_FILE="/opt/dlami/nvme/DMLAB/shcho/torchtitan/logs/h200/1024_llama8b/eval/eval_basemodel.sh" # "$(realpath "${BASH_SOURCE[0]}")"
LOG_DIR="$(dirname "${THIS_FILE}")"

CHECKPOINT_ROOT="/opt/dlami/nvme/DMLAB/shcho/torchtitan_data/base_model"
BASENAME_LIST=( # You can expand this list as needed
    "Llama-3.1-8B-Instruct"
) 

for BASENAME in "${BASENAME_LIST[@]}"; do

    OUTPUT_FILE="${LOG_DIR}/eval_${TODAY}_${BASENAME}.log"
    MODEL_PATH="${CHECKPOINT_ROOT}/${BASENAME}"
    RESULT_FILE="${MODEL_PATH}/eval_${TODAY}_${BASENAME}.json"

    # Check if the model path exists
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "❌Model path ${MODEL_PATH} not found — skipping." | tee -a "${OUTPUT_FILE}"
        continue
    fi

    # Print the current timestamp and the server name
    {
        echo -e "\n❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️"
        echo -e "✔️Current Timestamp: $(date)"
        echo -e "✔️SERVER: $(hostname) ($(hostname -I | awk '{print $1}')),  GPUs: ${CUDA_VISIBLE_DEVICES}"
        echo -e "✔️SCRIPT: ${THIS_FILE}"
        echo -e "✔️OUTPUT: ${OUTPUT_FILE}"
        echo -e "✔️RESULT: ${RESULT_FILE}"
        echo -e "✔️${EXPLAIN}"
        echo -e "☑️> python3 -m timelyfreeze.eval_hf_checkpoint --model_path=${MODEL_PATH} --dtype=float16 --model_type=${BASENAME} --batch_size=16 --device_map=cuda --output_json=${RESULT_FILE}"
        echo -e "❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️"
    } | tee -a ${OUTPUT_FILE}

    python3 -m timelyfreeze.eval_hf_checkpoint \
        --model_path=${MODEL_PATH} --output_json=${RESULT_FILE} \
        --dtype=float16 --model_type=${BASENAME} --batch_size=16 --device_map=cuda \
        2>&1 | tee -a ${OUTPUT_FILE}

done

echo "✅ Evaluation completed for all models. Logs saved in ${LOG_DIR}."