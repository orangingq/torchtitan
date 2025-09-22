#!/usr/bin/bash

# Define common environment variables
EXPLAIN="Main Table Experiment for Llama 3.1 8B on Runpod with 8 GPUs"
EXPERIMENT_TAG="0922_main"
TODAY="0922"

export WANDB_TAG="${EXPERIMENT_TAG}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_P2P_DISABLE=0 
export NCCL_IB_DISABLE=1 # no NVLink
export NCCL_LAUNCH_MODE=GROUP
export OMP_NUM_THREADS=1
export LOG_RANK=0,6,7
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHFT_LIGHTHOUSE="http://localhost:29510"
NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | grep -c .)

LOG_DIR="/workspace/torchtitan/logs/runpod8/${WANDB_TAG}"
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
)

for PP_SCHEDULER in gpipe ; do # gpipe 1f1b interleaved1f1b  interleavedzb zbv 
    for METRIC_TYPE in fullrand6 ; do  # fullrand6 apf nofreeze timelyapf

        OUTPUT_FILE="${LOG_DIR}/${TODAY}_${PP_SCHEDULER}_${METRIC_TYPE}.ans" # to support ANSI color conversion
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
                "--freezing.aggressiveness=0.05"
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

echo "All runs completed. Logs saved in ${LOG_DIR}."