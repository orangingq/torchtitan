#!/usr/bin/bash
#SBATCH --job-name=llama8b_train
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/h200/1104_llama8b/slurm-%j.out

bash ./logs/h200/1104_llama8b/run_1128.sh "$@"