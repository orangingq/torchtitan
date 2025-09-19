#!/usr/bin/bash
# set -ex

# Define common environment variables
EXPLAIN="Main Table Experiment"
CREATE_DATE="0820"
EXPERIMENT_TAG="main"
TODAY=$(date +%m%d)

export WANDB_TAG="${CREATE_DATE}_${EXPERIMENT_TAG}"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_P2P_DISABLE=1 # Not using NVLink
export OMP_NUM_THREADS=1
export LOG_RANK=0,3
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHFT_LIGHTHOUSE="http://localhost:29510"
NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | grep -c .)

LOG_DIR="/home/shcho/torchtitan/logs/dmserver4/${WANDB_TAG}"
CONFIG_FILE="${LOG_DIR}/config.toml"

COMMON_ARGS=(
    "--standalone"
    "--nnodes=1"
    "--nproc_per_node=${NGPU}"
    "--local-ranks-filter=${LOG_RANK}"
    "--role=rank"
    "--tee=3"
    "-m" 
    "timelyfreeze.train" 
    "--job.config_file=${CONFIG_FILE}"
    "--job.description=\"${EXPLAIN}\""
    "--training.global_batch_size=128" # gradient_accumulation_step = global_batch_size // local_batch_size
    "--training.local_batch_size=8" 
    "--parallelism.pipeline_parallel_microbatch_size=1" # num_microbatches = local_batch_size // microbatch_size
    "--training.seq_len=1024"
    "--training.steps=1000"
    "--parallelism.pipeline_parallel_degree=${NGPU}"
)

for PP_SCHEDULER in gpipe 1f1b interleaved1f1b ; do # 1f1b gpipe interleaved1f1b  interleavedzb zbv 
    for METRIC_TYPE in fullrand6 apf ; do # nofreeze

        OUTPUT_FILE="${LOG_DIR}/${TODAY}_${PP_SCHEDULER}_${METRIC_TYPE}.log"
        BASENAME="${TODAY}_${PP_SCHEDULER}_${METRIC_TYPE}_dm4"
        ADDITIONAL_ARGS=(
            "--parallelism.pipeline_parallel_schedule=${PP_SCHEDULER}" 
            "--metrics.basename=${BASENAME}"
            "--metrics.log_file=${OUTPUT_FILE}"
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
            printf "Current Timestamp: %s\n" "$(date)"
            printf "SERVER: %s (%s),  GPUs: %s\n" "$(hostname)" "$(hostname -I | awk '{print $1}')" "${CUDA_VISIBLE_DEVICES}"
            printf "Bash Script Start... %s\n" "${EXPLAIN}"
            printf "  Running with %s x %s ...\n" "${METRIC_TYPE}" "${PP_SCHEDULER}"
            printf "  Output -> %s\n" "${OUTPUT_FILE}"
            printf "> torchrun"
                for arg in "${COMMON_ARGS[@]}"; do printf " %s" "$arg"; done
                for arg in "${ADDITIONAL_ARGS[@]}"; do printf " %s" "$arg"; done
                for arg in "${FREEZE_ARGS[@]}"; do printf " %s" "$arg"; done
            printf "\n"
        } | tee -a "${OUTPUT_FILE}"

        torchrun "${COMMON_ARGS[@]}" "${ADDITIONAL_ARGS[@]}" "${FREEZE_ARGS[@]}"  2>&1 | tee -a ${OUTPUT_FILE}

    done
done

echo "All runs completed. Logs saved in ${LOG_DIR}."