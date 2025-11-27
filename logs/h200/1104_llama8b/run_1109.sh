#!/usr/bin/bash

# Define common environment variables
EXPLAIN="Llama 3.1 8B Experiment
1109: 1) resolved timelyapf issue, 2) seed fixed to 42 for all runs, 3) 3000 steps -> 2000 steps (better benchmark scores!)
1116: everything same, but with seed=11
"
EXPERIMENT_TAG="1104_llama8b"
TODAY="1109"

export WANDB_TAG="${EXPERIMENT_TAG}"
# Respect Slurm's CUDA_VISIBLE_DEVICES
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="${1:-0,1,2,3}"
    echo "✔️ Using manually set GPU(s): ${CUDA_VISIBLE_DEVICES}"
else
    echo "✔️ SLURM JOB GPUS: ${SLURM_JOB_GPUS}"
    echo "✔️ Using Slurm-assigned GPU(s): ${CUDA_VISIBLE_DEVICES}"
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LOG_RANK=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHFT_LIGHTHOUSE="http://localhost:29510"
NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | grep -c .)

LOG_DIR="/opt/dlami/nvme/DMLAB/shcho/torchtitan/logs/h200/${EXPERIMENT_TAG}"
CONFIG_FILE="${LOG_DIR}/config_${TODAY}.toml"
THIS_FILE="${LOG_DIR}/run.sh" # "$(realpath "${BASH_SOURCE[0]}")"

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

for PP_SCHEDULER in GPipe 1F1B ; do # GPipe 1F1B Interleaved1F1B  InterleavedZeroBubble ZBVZeroBubble
    for METRIC_TYPE in apf timelyapf ; do #nofreeze apf auto fullrand7 timelyapf timelyauto 

        OUTPUT_FILE="${LOG_DIR}/${TODAY}_${PP_SCHEDULER}_${METRIC_TYPE}_th3e-3.log"
        BASENAME="${TODAY}_${PP_SCHEDULER}_${METRIC_TYPE}_th3e-3_h200"

        # Skip evaluation if result file already exists
        if [ -f "${OUTPUT_FILE}" ]; then
            echo "⚠️Result file ${OUTPUT_FILE} already exists — skipping evaluation." 
            continue
        fi

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
                "--freezing.threshold=0.003" # 0.05 -> 0.005 (freezing more)
                "--freezing.max_freeze_ratio=0.7"
                "--freezing.percentile=80" # 70 -> 80 (freezing more)
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