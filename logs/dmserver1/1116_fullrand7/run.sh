#!/usr/bin/bash

# Define common environment variables
EXPLAIN="TimelyFreeze (fullrand7) configuration experiment on different max_freeze_ratio values."

EXPERIMENT_TAG="1116_fullrand7"
TODAY="1116"

export WANDB_TAG="${EXPERIMENT_TAG}"
export CUDA_VISIBLE_DEVICES=1,2,3,4
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

SEED=42
for PP_SCHEDULER in 1F1B ; do # GPipe 1F1B Interleaved1F1B  InterleavedZeroBubble ZBVZeroBubble
    for MAX_FREEZE_RATIO in 0.4 0.5 0.6 0.7 0.8 0.9 ; do # 0.01 0.03 0.05 0.07 0.1 0.2 0.3 0.4 0.5

        OUTPUT_FILE="${LOG_DIR}/${TODAY}_${PP_SCHEDULER}_fullrand7_mfr${MAX_FREEZE_RATIO}_${SEED}.log"
        BASENAME="${TODAY}_${PP_SCHEDULER}_fullrand7_mfr${MAX_FREEZE_RATIO}_${SEED}_dm1"

        # Skip evaluation if result file already exists
        if [ -f "${OUTPUT_FILE}" ]; then
            echo "⚠️Result file ${OUTPUT_FILE} already exists — skipping evaluation." 
            continue
        fi

        ADDITIONAL_ARGS=(
            "--parallelism.pipeline_parallel_schedule=${PP_SCHEDULER}" 
            "--job.basename=${BASENAME}"
            "--training.seed=${SEED}"
            "--training.local_batch_size=16" # only for GPipe
            "--parallelism.pipeline_parallel_microbatch_size=2" # only for GPipe
        )
        FREEZE_ARGS=(
            "--freezing.freeze"
            "--freezing.metric_type=fullrand7"
            # "--freezing.threshold=0.01"
            # "--freezing.percentile=80"
            "--freezing.max_freeze_ratio=${MAX_FREEZE_RATIO}"
        )

        # Print the current timestamp and the server name
        {
            echo -e "\n🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥"
            echo -e "✔️Current Timestamp: $(date)"
            echo -e "✔️SERVER: $(hostname) ($(hostname -I | awk '{print $1}')),  GPUs: ${CUDA_VISIBLE_DEVICES}"
            echo -e "✔️SCRIPT: ${THIS_FILE}"
            echo -e "✔️OUTPUT: ${OUTPUT_FILE}"
            echo -e "✔️${EXPLAIN}"
            echo -e "✔️Running with fullrand7 (max_freeze_ratio=${MAX_FREEZE_RATIO}) x ${PP_SCHEDULER} ... "
            echo -e "☑️> torchrun ${COMMON_ARGS[@]} ${PP_ARGS[@]} ${FREEZE_ARGS[@]}"
            echo -e "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥"
        } | tee -a ${OUTPUT_FILE}

        torchrun "${COMMON_ARGS[@]}" "${ADDITIONAL_ARGS[@]}" "${FREEZE_ARGS[@]}"  2>&1 | tee -a ${OUTPUT_FILE}

    done
done

echo "✅ All runs completed. Logs saved in ${LOG_DIR}."