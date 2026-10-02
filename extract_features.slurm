#!/bin/bash
#SBATCH --job-name=PGMT_FeatureExtract
#SBATCH --partition=d
#SBATCH --nodelist=oniwaka
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=70G
#SBATCH --time=48:00:00
#SBATCH --output=logs/extract_features_%j.log

# Enforce clean thread allocation
export OMP_NUM_THREADS=1

# Route caches away from restricted /home/ directory to NAS
export HF_HOME="/pnr/lab/nurul/PGMT_Code/hf_cache"
export TORCH_HOME="/pnr/lab/nurul/PGMT_Code/torch_cache"

echo "=========================================================="
echo "Starting PGMT Feature Extraction on Node: $SLURMD_NODENAME"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "Target NAS Output: /pnr/lab/nurul/PGMT_Code/offline_features"
echo "=========================================================="

# Environment activation
source /shared/miniconda3/etc/profile.d/conda.sh
conda activate /pnr/lab/nurul/PGMT_Code/env

# Ensure destination directory exists on NAS
mkdir -p /pnr/lab/nurul/PGMT_Code/offline_features
mkdir -p logs

# Run extraction using standard single-GPU Python execution
python extract_offline_features.py

echo "=========================================================="
echo "Feature Extraction Job Complete."
echo "=========================================================="