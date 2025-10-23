#!/usr/bin/bash

# Define common environment variables
EXPLAIN="Main Table Experiment"
EXPERIMENT_TAG="1020_main"
TODAY="1020"

export WANDB_TAG="${EXPERIMENT_TAG}"
export CUDA_VISIBLE_DEVICES="${1:-0}"
echo "✔️Using GPU ${CUDA_VISIBLE_DEVICES}"

THIS_FILE="$(realpath "${BASH_SOURCE[0]}")"
LOG_DIR="/home/shcho/torchtitan/logs/dmserver1/${WANDB_TAG}/eval"

for PP_SCHEDULER in 1f1b ; do # 1f1b gpipe interleaved1f1b  interleavedzb zbv 
    for METRIC_TYPE in nofreeze  ; do # nofreeze fullrand7

        OUTPUT_FILE="${LOG_DIR}/${TODAY}_${PP_SCHEDULER}_${METRIC_TYPE}.log"
        BASENAME="${TODAY}_${PP_SCHEDULER}_${METRIC_TYPE}_dm1"
        MODEL_DIR="/data2/shcho/torchtitan/checkpoint/${BASENAME}/step-500"
        MODEL_PATH="${MODEL_DIR}/sharded"
        RESULT_FILE="${MODEL_PATH}/${TODAY}_${PP_SCHEDULER}_${METRIC_TYPE}.json"    

        # Check if the model path exists
        if [ ! -d "${MODEL_DIR}" ]; then
            echo "❌Model directory ${MODEL_DIR} not found — skipping." | tee -a "${OUTPUT_FILE}"
            continue
        fi

        # Convert to HuggingFace format if .safetensors files are not found
        if ! ls "${MODEL_PATH}"/*.safetensors 1> /dev/null 2>&1; then
          echo "➡️No .safetensors files found in ${MODEL_PATH}. Converting to HuggingFace format..." | tee -a "${OUTPUT_FILE}"
          torchrun --nproc_per_node=1 --nnodes=1 --standalone --local_addr=127.0.0.1 --role=rank --tee=3 -m scripts.checkpoint_conversion.convert_to_hf \
            "${MODEL_DIR}" "${MODEL_DIR}" --model_name=llama3 --model_flavor=1B
        fi

        # Ensure index file exists
        echo "➡️Recreating Index file at ${MODEL_PATH}/model.safetensors.index.json." | tee -a "${OUTPUT_FILE}"
        cat > "${MODEL_PATH}/model.safetensors.index.json" <<'JSON'
{
  "metadata": {
    "total_size": 961583888
  },
  "weight_map": {
    "model.layers.0.weight": "shard-00001-model-00001-of-00005.safetensors",
    "model.layers.1.weight": "shard-00001-model-00002-of-00005.safetensors",
    "model.layers.2.weight": "shard-00001-model-00003-of-00005.safetensors",
    "model.layers.3.weight": "shard-00001-model-00004-of-00005.safetensors",
    "model.layers.4.weight": "shard-00001-model-00005-of-00005.safetensors"
  }
}
JSON


        # Print the current timestamp and the server name
        {
            echo -e "\n🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥"
            echo -e "✔️Current Timestamp: $(date)"
            echo -e "✔️SERVER: $(hostname) ($(hostname -I | awk '{print $1}')),  GPUs: ${CUDA_VISIBLE_DEVICES}"
            echo -e "✔️SCRIPT: ${THIS_FILE}"
            echo -e "✔️OUTPUT: ${OUTPUT_FILE}"
            echo -e "✔️${EXPLAIN}"
            echo -e "✔️Running with ${METRIC_TYPE} x ${PP_SCHEDULER} ... "
            echo -e "☑️> torchrun ${COMMON_ARGS[@]} ${PP_ARGS[@]} ${FREEZE_ARGS[@]}"
            echo -e "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥"
        } | tee -a ${OUTPUT_FILE}

        python3 -m timelyfreeze.eval_hf_checkpoint --model_path=${MODEL_PATH} \
          --dtype=float16 --model_type=Llama-3.2-1B --batch_size=16 --device_map=cuda \
          --output_json=${RESULT_FILE} 2>&1 | tee -a ${OUTPUT_FILE}

    done
done

echo "All evaluations completed. Logs saved in ${LOG_DIR}."