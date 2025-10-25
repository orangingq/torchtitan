#!/usr/bin/bash
#SBATCH --job-name=llama8b_train
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/h200/1024_llama8b/slurm-%j.out

# Define common environment variables
EXPLAIN="Llama 3.1 8B Instruct Experiment"
EXPERIMENT_TAG="1024_llama8b"
TODAY="1024"

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
# export NCCL_P2P_DISABLE=1 # Not using NVLink
# export OMP_NUM_THREADS=1
export LOG_RANK=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHFT_LIGHTHOUSE="http://localhost:29510"
NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | grep -c .)

LOG_DIR="/opt/dlami/nvme/DMLAB/shcho/torchtitan/logs/h200/${WANDB_TAG}"
CONFIG_FILE="${LOG_DIR}/config.toml"
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

for PP_SCHEDULER in GPipe ; do # GPipe 1F1B Interleaved1F1B  InterleavedZeroBubble ZBVZeroBubble
    for METRIC_TYPE in fullrand7 apf nofreeze auto timelyapf ; do  # 

        OUTPUT_FILE="${LOG_DIR}/${TODAY}_${PP_SCHEDULER}_${METRIC_TYPE}.log"
        BASENAME="${TODAY}_${PP_SCHEDULER}_${METRIC_TYPE}_h200"
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