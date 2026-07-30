#!/bin/bash
#SBATCH --job-name=PGMT_Xsub_Eval
#SBATCH --exclude=antelope                   
#SBATCH --gres=gpu:1                         
#SBATCH --cpus-per-task=8             
#SBATCH --mem=70G                                   
#SBATCH --output=logs/pgmt_xsub_eval_%j.log  

# Silence PyTorch warnings
export OMP_NUM_THREADS=1

# Force local caches to bypass /home/ permission drops
export HF_HOME="/pnr/lab/nurul/PGMT_Code/hf_cache"
export TORCH_HOME="/pnr/lab/nurul/PGMT_Code/torch_cache"

echo "=========================================================="
echo "Running PGMT Phase 1 X-Sub Evaluation on Node: $SLURMD_NODENAME"
echo "=========================================================="

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate /pnr/lab/nurul/PGMT_Code/env

python eval_retrieval_xsub.py

echo "Evaluation Complete."