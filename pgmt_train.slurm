#!/bin/bash
#SBATCH --job-name=PGMT_Stage1
#SBATCH --partition=d
#SBATCH --nodelist=oniwaka            
#SBATCH --gres=gpu:2                  
#SBATCH --cpus-per-task=8             
#SBATCH --mem=60G                     
#SBATCH --time=48:00:00               
#SBATCH --output=pgmt_training_%j.log 

echo "=========================================================="
echo "Diagnostics for Node: $SLURMD_NODENAME"
nvidia-smi
echo "Checking PyTorch/CUDA visibility:"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Driver Version: {torch.version.cuda}')"
echo "=========================================================="

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate /pnr/lab/nurul/PGMT_Code/env

torchrun --nproc_per_node=2 train_gta_fa.py
