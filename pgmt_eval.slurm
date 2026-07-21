#!/bin/bash
#SBATCH --job-name=PGMT_Eval
#SBATCH --partition=d
#SBATCH --nodelist=imawaka
#SBATCH --gres=gpu:1                         
#SBATCH --cpus-per-task=8             
#SBATCH --mem=90G                            
#SBATCH --time=02:00:00           
#SBATCH --output=logs/pgmt_eval_%j.log

# Silence PyTorch warnings 
export OMP_NUM_THREADS=1

# Force local caches to bypass /home/ permission
export HF_HOME="/pnr/lab/nurul/PGMT_Code/hf_cache"
export TORCH_HOME="/pnr/lab/nurul/PGMT_Code/torch_cache"

echo "=========================================================="
echo "Running PGMT Phase 1 Evaluation on Node: $SLURMD_NODENAME"
echo "=========================================================="

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate /pnr/lab/nurul/PGMT_Code/env

# Run the evaluation script
python eval_retrieval.py

echo "Evaluation Complete."