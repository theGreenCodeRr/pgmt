#!/bin/bash
#SBATCH --job-name=PGMT_SplitTrain
#SBATCH --partition=d
#SBATCH --nodelist=oniwaka
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=70G
#SBATCH --time=48:00:00
#SBATCH --output=logs/pgmt_split_training_%j.log

# Terminate immediately if any command fails
set -e

export OMP_NUM_THREADS=1

echo "=========================================================="
echo "Starting Tier 3 Multi-Split Retraining Pipeline"
echo "Target Node: $SLURMD_NODENAME"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate /pnr/lab/nurul/PGMT_Code/env

# RUN 1: Standard Inductive X-Sub
echo "[*] Launching Protocol: X-Sub | ZSL: 0"
python train_split.py --protocol xsub --zsl_classes 0 --batch_size 256

# RUN 2: Standard Inductive X-Set
echo "[*] Launching Protocol: X-Set | ZSL: 0"
python train_split.py --protocol xset --zsl_classes 0 --batch_size 256

# RUN 3: Zero-Shot 110/10
echo "[*] Launching Protocol: X-Sub | ZSL: 10"
python train_split.py --protocol xsub --zsl_classes 10 --batch_size 256

# RUN 4: Zero-Shot 96/24
echo "[*] Launching Protocol: X-Sub | ZSL: 24"
python train_split.py --protocol xsub --zsl_classes 24 --batch_size 256

echo "=========================================================="
echo "All 4 Retraining Protocols Successfully Completed."
echo "=========================================================="
