#!/usr/bin/bash

# Define common environment variables
EXPLAIN="Main Table Experiment, without streaming mode, sample-level with truncation, 2 epochs, with bf16 autocast
+ more freezing in autofreeze mode. 
+ lr_scheduler min_lr = 0 -> 1e-6, cosine decay.
"
EXPERIMENT_TAG="1029_llama1b"
TODAY="1031"

export WANDB_TAG="${EXPERIMENT_TAG}"
export CUDA_VISIBLE_DEVICES=3,4,5,6
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_P2P_DISABLE=1 # Not using NVLink
export OMP_NUM_THREADS=1
export LOG_RANK=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHFT_LIGHTHOUSE="http://localhost:29510"
NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | grep -c .)

THIS_FILE="$(realpath "${BASH_SOURCE[0]}")"
LOG_DIR="$(dirname "${THIS_FILE}")"
CONFIG_FILE="${LOG_DIR}/config_${TODAY}.toml"

COMMON_ARGS=(
    "--standalone"
    "--nnodes=1"
    "--nproc_per_node=${NGPU}"
    "--local_addr=127.0.0.1"
    "--local-ranks-filter=${LOG_RANK}"
    "--role=rank"
    "--tee=3"
    "-m" 
    "timelyfreeze.train" 
    "--job.config_file=${CONFIG_FILE}"
    "--job.description=\"${EXPLAIN}\""
    "--parallelism.pipeline_parallel_degree=${NGPU}"
)

EXPERIMENT_LIST=( # You can expand this list as needed
#   "GPipe auto"
#   "GPipe timelyapf"
#   "1F1B timelyauto"
#   "1F1B timelyapf"
#   "1F1B fullrand7"
#   "1F1B auto"
#   "Interleaved1F1B fullrand7"
#   "Interleaved1F1B timelyauto"
#   "Interleaved1F1B timelyapf"
#   "Interleaved1F1B apf"
)

for PP_SCHEDULER in GPipe 1F1B Interleaved1F1B ; do # 1F1B GPipe Interleaved1F1B  InterleavedZeroBubble ZBVZeroBubble
    for METRIC_TYPE in nofreeze apf auto fullrand7 timelyapf timelyauto  ; do # nofreeze apf auto fullrand7 timelyapf timelyauto
# for EXPERIMENT in "${EXPERIMENT_LIST[@]}"; do
#     IFS=' ' read -r -a EXP_ARRAY <<< "$EXPERIMENT"
#     PP_SCHEDULER="${EXP_ARRAY[0]}"
#     METRIC_TYPE="${EXP_ARRAY[1]}"

        OUTPUT_FILE="${LOG_DIR}/${TODAY}_${PP_SCHEDULER}_${METRIC_TYPE}.log"
        BASENAME="${TODAY}_${PP_SCHEDULER}_${METRIC_TYPE}_dm1"
        ADDITIONAL_ARGS=(
            "--parallelism.pipeline_parallel_schedule=${PP_SCHEDULER}" 
            "--job.basename=${BASENAME}"
        )
        if [[ "$METRIC_TYPE" == "nofreeze" ]]; then       
            FREEZE_ARGS=(
                "--freezing.no-freeze"
            )
        else
            FREEZE_ARGS=(
                "--freezing.freeze"
                "--freezing.metric_type=${METRIC_TYPE}"
            )
        fi

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

        torchrun "${COMMON_ARGS[@]}" "${ADDITIONAL_ARGS[@]}" "${FREEZE_ARGS[@]}"  2>&1 | tee -a ${OUTPUT_FILE}

    done
done

echo "✅ All runs completed. Logs saved in ${LOG_DIR}."