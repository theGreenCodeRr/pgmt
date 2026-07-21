#!/bin/bash
#SBATCH --job-name=PGMT_GTA_FA
#SBATCH --partition=d
#SBATCH --nodelist=oniwaka
#SBATCH --requeue                     # AUTO-RESTART: Automatically requeue if the node crashes
#SBATCH --gres=gpu:2                
#SBATCH --cpus-per-task=8            
#SBATCH --mem=70G
#SBATCH --time=48:00:00               
#SBATCH --output=logs/pgmt_training_%j.log   

export OMP_NUM_THREADS=1

echo "=========================================================="
echo "Starting PGMT Phase 1 Training on Node: $SLURMD_NODENAME"
echo "=========================================================="

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate /pnr/lab/nurul/PGMT_Code/env

# Launch the Distributed Data Parallel script across GPUs
torchrun --nproc_per_node=2 train_gta_fa.py 

echo "Training Complete."