#!/bin/bash
#SBATCH --job-name=PGMT_ZSL_Splits
#SBATCH --nodes=1
#SBATCH --nodelist=imawaka,bateira
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8     
#SBATCH --mem=50G          
#SBATCH --output=logs/pgmt_zsl_splits_%j.log   

export OMP_NUM_THREADS=1
export HF_HOME="/pnr/lab/nurul/PGMT_Code/hf_cache"
export TORCH_HOME="/pnr/lab/nurul/PGMT_Code/torch_cache"

echo "=========================================================="
echo "Running PGMT Zero-Shot Target Splits on Node: $SLURMD_NODENAME"
echo "=========================================================="

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate /pnr/lab/nurul/PGMT_Code/env

echo "[*] Evaluating 110/10 Split..."
python eval_zeroshot_split.py --split 110_10

echo "[*] Evaluating 96/24 Split..."
python eval_zeroshot_split.py --split 96_24

echo "Evaluations Complete."
