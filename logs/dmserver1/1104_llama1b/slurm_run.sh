#!/usr/bin/bash
#SBATCH --job-name=llama1b_train
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/dmserver1/1104_llama1b/slurm-%j.out

bash ./logs/dmserver1/1104_llama1b/run.sh "$@"