#!/bin/bash
#SBATCH --job-name=PGMT_Linear
#SBATCH --nodes=1
#SBATCH --nodelist=imawaka,bateira
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8     
#SBATCH --mem=50G
#SBATCH --output=logs/pgmt_linear_probe_%j.log  

export OMP_NUM_THREADS=1
export HF_HOME="/pnr/lab/nurul/PGMT_Code/hf_cache"
export TORCH_HOME="/pnr/lab/nurul/PGMT_Code/torch_cache"

echo "=========================================================="
echo "Running PGMT Phase 1 Linear Probing on Node: $SLURMD_NODENAME"
echo "=========================================================="

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate /pnr/lab/nurul/PGMT_Code/env

echo "[*] Starting X-Sub Linear Probe..."
python eval_linear.py --protocol xsub --epochs 100

echo "[*] Starting X-View Linear Probe..."
python eval_linear.py --protocol xview --epochs 100

echo "Evaluation Complete."