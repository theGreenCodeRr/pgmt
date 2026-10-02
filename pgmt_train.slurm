#!/bin/bash
#SBATCH --job-name=PGMT_GTA_FA
#SBATCH --partition=d
#SBATCH --nodelist=imawaka
#SBATCH --requeue
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=70G
#SBATCH --output=logs/pgmt_training_old_%j.log

export OMP_NUM_THREADS=1

# Force local caches to bypass /home/ permission drops
export HF_HOME="/pnr/lab/nurul/PGMT_Code/hf_cache"
export TORCH_HOME="/pnr/lab/nurul/PGMT_Code/torch_cache"

echo "=========================================================="
echo "Starting OLD PGMT Phase 1 Training on Node: $SLURMD_NODENAME"
echo "=========================================================="

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate /pnr/lab/nurul/PGMT_Code/env

# Launch the Distributed Data Parallel script across GPUs
torchrun --nproc_per_node=2 train_gta_faOld.py 

echo "Training Complete."
