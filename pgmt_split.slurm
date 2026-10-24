#!/bin/bash
#SBATCH --job-name=Split_ZSL0_xset
#SBATCH --partition=d
#SBATCH --nodes=1
#SBATCH --nodelist=imawaka,oniwaka
#SBATCH --gres=gpu:2              
#SBATCH --cpus-per-task=16
#SBATCH --mem=70G                # or 70G for oniwaka
#SBATCH --output=logs/split_zsl0_%j.log

set -e

# ======================================================================
# --- [ZONE 2] TRAINING HYPERPARAMETERS 
# ======================================================================
# RUN 1: Standard Inductive X-Sub (All 120 Classes)
#PROTOCOL="xsub"
#ZSL_CLASSES=0

# RUN 2: Standard Inductive X-Set (All 120 Classes)
PROTOCOL="xset"
ZSL_CLASSES=0

# RUN 3: Zero-Shot X-Sub (110 Seen / 10 Unseen)
# PROTOCOL="xsub"
# ZSL_CLASSES=10

# RUN 4: Zero-Shot X-Sub (96 Seen / 24 Unseen)
# PROTOCOL="xsub"
# ZSL_CLASSES=24

BATCH_SIZE=256            
EPOCHS=250

# ======================================================================
# --- [ZONE 3] NETWORK & STORAGE PATHS ---
# ======================================================================
# Old NAS path for staging data (Read-Only)
BASE_ENV_DIR="/pnr/lab/nurul/PGMT_Code"
NAS_SKELETON_DIR="/pnr/lab/nurul/datasets/ntu_rgbd/skeleton/"
NAS_FEATURE_DIR="/pnr/lab/nurul/PGMT_Code/offline_features/"

# NAS path for saving checkpoints (Write-Only)
NAS_SAVE_DIR="/pnr/lab/nurul/PGMT_Code/checkpoints_retrain"

# ======================================================================
# EXECUTION LOGIC (Do not modify below this line)
# ======================================================================
LOCAL_SKEL_DIR="/tmp/skeleton_cache"
LOCAL_PT_DIR="/tmp/feature_cache"

export OMP_NUM_THREADS=1
export HF_HOME="${BASE_ENV_DIR}/hf_cache"
export TORCH_HOME="${BASE_ENV_DIR}/torch_cache"
export PATH="/pnr/lab/nurul/PGMT_Code/env/bin:$PATH"   # if /shared conda broken, use this path instead

echo "=========================================================="
echo "Starting DDP Split Training on Node: $SLURMD_NODENAME"
echo "Protocol: $PROTOCOL | ZSL Classes: $ZSL_CLASSES | Local Batch: $BATCH_SIZE"
echo "Target Checkpoint Directory: $NAS_SAVE_DIR"
echo "=========================================================="

echo "[*] Verifying persistent local NVMe caches on /tmp..."
mkdir -p $LOCAL_SKEL_DIR
mkdir -p $LOCAL_PT_DIR

echo "[*] Staging Skeletons from: $NAS_SKELETON_DIR"
rsync -a ${NAS_SKELETON_DIR} ${LOCAL_SKEL_DIR}/

echo "[*] Staging Features from: $NAS_FEATURE_DIR"
rsync -a ${NAS_FEATURE_DIR} ${LOCAL_PT_DIR}/

echo "[*] Activating Conda Environment..."
# source /shared/miniconda3/etc/profile.d/conda.sh
# conda activate ${BASE_ENV_DIR}/env

echo "[*] Launching Distributed PyTorch Training..."
torchrun --standalone --nproc_per_node=2 train_split.py \
    --protocol $PROTOCOL \
    --zsl_classes $ZSL_CLASSES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr 3e-4 \
    --save_dir $NAS_SAVE_DIR

echo "=========================================================="
echo "Job Successfully Completed on Node: $SLURMD_NODENAME"
echo "=========================================================="
