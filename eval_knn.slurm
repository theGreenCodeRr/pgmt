#!/bin/bash
#SBATCH --job-name=PGMT_kNN
#SBATCH --nodes=1
#SBATCH --nodelist=imawaka,bateira,otowaka,xenon,ushiwaka
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8     
#SBATCH --mem=50G         
#SBATCH --output=logs/pgmt_knn_%j.log  

export OMP_NUM_THREADS=1   # Silence PyTorch warnings
export HF_HOME="/pnr/lab/nurul/PGMT_Code/hf_cache"
export TORCH_HOME="/pnr/lab/nurul/PGMT_Code/torch_cache"

echo "=========================================================="
echo "Running PGMT Phase 1 k-NN Classification on Node: $SLURMD_NODENAME"
echo "=========================================================="

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate /pnr/lab/nurul/PGMT_Code/env

echo "[*] Starting X-Sub Evaluation..."
python eval_knn.py --protocol xsub

echo "[*] Starting X-View Evaluation..."
python eval_knn.py --protocol xview

echo "Evaluation Complete."
