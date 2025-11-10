#!/usr/bin/bash

# Define common environment variables
EXPLAIN="Llama 3.2 1B Base Model Evaluation"
TODAY="1109"

export CUDA_VISIBLE_DEVICES="${1:-0}"
echo "✔️Using GPU ${CUDA_VISIBLE_DEVICES}"

THIS_FILE="$(realpath "${BASH_SOURCE[0]}")"
LOG_DIR="$(dirname "${THIS_FILE}")"

CHECKPOINT_ROOT="/data2/shcho/torchtitan/base_model"
BASENAME_LIST=( # You can expand this list as needed
  "Llama-3.2-1B"
) 
TASKS="mmlu,hellaswag,arc_challenge,truthfulqa_mc1"

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
        echo -e "☑️> python3 -m timelyfreeze.evaluation --model_path=${MODEL_PATH} --dtype=float16 --model_type=${BASENAME} --batch_size=16 --device_map=cuda --tasks=${TASKS} --num_fewshot 0 --num_fewshot_task mmlu=5,arc_challenge=25 --output_json=${RESULT_FILE}"
        echo -e "❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️"
    } | tee -a ${OUTPUT_FILE}

    python3 -m timelyfreeze.evaluation \
        --model_path=${MODEL_PATH} --output_json=${RESULT_FILE} \
        --dtype=float16 --model_type=${BASENAME} --batch_size=16 --device_map=cuda --tasks=${TASKS} \
        --num_fewshot 0 --num_fewshot_task mmlu=5,arc_challenge=10 \
        2>&1 | tee -a ${OUTPUT_FILE}


done

echo "✅ Evaluation completed for all models. Logs saved in ${LOG_DIR}."