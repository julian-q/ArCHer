#!/bin/bash
#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --nodelist=sphinx5
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

python run.py --config-name archer_gemm_stable

