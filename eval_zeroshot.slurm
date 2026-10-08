#!/bin/bash
#SBATCH --job-name=PGMT_ZeroShot
#SBATCH --nodes=1
#SBATCH --nodelist=bateira,otowaka,xenon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8     
#SBATCH --mem=25G          
#SBATCH --output=logs/pgmt_ZeroShot_%j.log  

export OMP_NUM_THREADS=1   # Silence PyTorch warnings
export HF_HOME="/pnr/lab/nurul/PGMT_Code/hf_cache"
export TORCH_HOME="/pnr/lab/nurul/PGMT_Code/torch_cache"

echo "=========================================================="
echo "Running PGMT Zero-Shot Text Retrieval on Node: $SLURMD_NODENAME"
echo "=========================================================="

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate /pnr/lab/nurul/PGMT_Code/env

echo "[*] Starting 120-Class Open-Vocabulary Evaluation..."
python eval_zeroshot.py --protocol all --split all

echo "Evaluation Complete."
