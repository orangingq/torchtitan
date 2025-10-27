#!/usr/bin/bash

# Define common environment variables
EXPLAIN="Main Table Experiment"
EXPERIMENT_TAG="1026_llama1b"

export CUDA_VISIBLE_DEVICES="${1:-0}"
echo "✔️Using GPU ${CUDA_VISIBLE_DEVICES}"

THIS_FILE="$(realpath "${BASH_SOURCE[0]}")"
LOG_DIR="$(dirname "${THIS_FILE}")"

CHECKPOINT_ROOT="/data3/shcho/torchtitan/checkpoint"
BASENAME_LIST=( # You can expand this list as needed
  "1026_GPipe_nofreeze_dm1"
  "1026_GPipe_fullrand7_dm1"
  "1026_GPipe_apf_dm1"
  "1026_GPipe_auto_dm1"
  "1026_GPipe_timelyapf_dm1"
  # "1026_GPipe_timelyauto_dm1"
) 
MODEL_TYPE="Llama-3.2-1B-Instruct"

for BASENAME in "${BASENAME_LIST[@]}"; do

    OUTPUT_FILE="${LOG_DIR}/eval_${BASENAME}.log"
    MODEL_PATH="${CHECKPOINT_ROOT}/${BASENAME}/step-1000"
    RESULT_FILE="${MODEL_PATH}/eval_${BASENAME}.json"

    # Check if the model path exists
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "❌Model directory ${MODEL_PATH} not found — skipping." | tee -a "${OUTPUT_FILE}"
        continue
    fi

    # Convert to HuggingFace format if .safetensors files are not found
    if ! ls "${MODEL_PATH}"/*.safetensors 1> /dev/null 2>&1; then
      echo "➡️No .safetensors files found in ${MODEL_PATH}. Converting to HuggingFace format..." | tee -a "${OUTPUT_FILE}"
      torchrun --nproc_per_node=1 --nnodes=1 --standalone --local_addr=127.0.0.1 --role=rank --tee=3 -m scripts.checkpoint_conversion.convert_to_hf \
        "${MODEL_PATH}" "${MODEL_PATH}" --model_name=llama3 --model_flavor=1B
    fi

    # Ensure index file exists
    if [ ! -f "${MODEL_PATH}/model.safetensors.index.json" ]; then
        echo "➡️Recreating Index file at ${MODEL_PATH}/model.safetensors.index.json." | tee -a "${OUTPUT_FILE}"
        cat > "${MODEL_PATH}/model.safetensors.index.json" <<'JSON'
{
  "metadata": {
    "total_size": 961583888
  },
  "weight_map": {
    "model.layers.0.weight": "model-00001-of-00005.safetensors",
    "model.layers.1.weight": "model-00002-of-00005.safetensors",
    "model.layers.2.weight": "model-00003-of-00005.safetensors",
    "model.layers.3.weight": "model-00004-of-00005.safetensors",
    "model.layers.4.weight": "model-00005-of-00005.safetensors"
  }
}
JSON
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
        echo -e "☑️> python3 -m timelyfreeze.eval_hf_checkpoint --model_path=${MODEL_PATH} --dtype=float16 --model_type=${MODEL_TYPE} --batch_size=32 --device_map=cuda --output_json=${RESULT_FILE}"
        echo -e "❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️❄️"
    } | tee -a ${OUTPUT_FILE}

    python3 -m timelyfreeze.eval_hf_checkpoint \
      --model_path=${MODEL_PATH} --output_json=${RESULT_FILE} \
      --dtype=float16 --model_type=${MODEL_TYPE} --batch_size=32 --device_map=cuda \
      2>&1 | tee -a ${OUTPUT_FILE}

done

echo "✅ Evaluation completed for all models. Logs saved in ${LOG_DIR}."