#!/bin/bash
#SBATCH --job-name=PGMT_Stage1
#SBATCH --partition=d                 # <-- Added the partition name
#SBATCH --nodelist=oniwaka            
#SBATCH --gres=gpu:2                  # <-- Using strict generic resource syntax
#SBATCH --cpus-per-task=8             
#SBATCH --mem=60G                     
#SBATCH --time=48:00:00               
#SBATCH --output=pgmt_training_%j.log 

echo "=========================================================="
echo "Starting PGMT Phase 1 Training on Node: $SLURMD_NODENAME"
echo "=========================================================="

# Launch the Distributed Data Parallel script
torchrun --nproc_per_node=2 train_stage1.py

echo "Training Complete."