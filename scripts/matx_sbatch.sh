#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

python run.py --config-name archer_gemm_stable

