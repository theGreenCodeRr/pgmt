#!/bin/bash
#SBATCH --job-name=Split_Eval_Fast
#SBATCH --partition=d
#SBATCH --nodelist=imawaka
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/pgmt_split_eval_%j.log

set -e

export OMP_NUM_THREADS=1
export HF_HOME="/pnr/lab/nurul/PGMT_Code/hf_cache"
export TORCH_HOME="/pnr/lab/nurul/PGMT_Code/torch_cache"
export PATH="/pnr/lab/nurul/PGMT_Code/env/bin:$PATH"   # Bypassing the broken global conda

echo "=========================================================="
echo "Starting Master Split Evaluation on Node: $SLURMD_NODENAME"
echo "=========================================================="

# ==========================================
# PERSISTENT LOCAL NVMe STAGING 
# ==========================================
LOCAL_SKEL_DIR="/tmp/skeleton_cache"
LOCAL_PT_DIR="/tmp/feature_cache"

echo "[*] Staging files to persistent local drive caches..."
mkdir -p $LOCAL_SKEL_DIR
mkdir -p $LOCAL_PT_DIR

# Using rsync to copy only missing files. Subsequent runs will take ~1s here.
rsync -a /pnr/lab/nurul/datasets/ntu_rgbd/skeleton/ $LOCAL_SKEL_DIR/
rsync -a /pnr/lab/nurul/PGMT_Code/offline_features/ $LOCAL_PT_DIR/
echo "[*] Local staging verified and complete!"


# ==========================================
# 1. Standard Inductive Evaluations
# ==========================================
echo "[*] Launching X-Sub Inductive Evaluation (k-NN & Linear Probe)"
python eval_inductive.py --protocol xsub \
    --checkpoint /pnr/lab/nurul/PGMT_Code/checkpoints_retrain/xsub_zsl0/epoch_170.pth \
    --skeleton_dir $LOCAL_SKEL_DIR \
    --offline_features_dir $LOCAL_PT_DIR

echo "[*] Launching X-Set Inductive Evaluation (k-NN & Linear Probe)"
python eval_inductive.py --protocol xset \
    --checkpoint /pnr/lab/nurul/PGMT_Code/checkpoints_retrain/xset_zsl0/epoch_170.pth \
    --skeleton_dir $LOCAL_SKEL_DIR \
    --offline_features_dir $LOCAL_PT_DIR

# ==========================================
# 2. Zero-Shot Alignment Evaluations (COMMENTED OUT FOR NOW)
# ==========================================
# echo "[*] Launching 110/10 Zero-Shot Evaluation"
# python eval_zeroshot_splits.py --split_type 110_10 \
#     --checkpoint /pnr/lab/nurul/PGMT_Code/checkpoints_retrain/xsub_zsl10/epoch_170.pth \
#     --skeleton_dir $LOCAL_SKEL_DIR \
#     --offline_features_dir $LOCAL_PT_DIR
#
# echo "[*] Launching 96/24 Zero-Shot Evaluation"
# python eval_zeroshot_splits.py --split_type 96_24 \
#     --checkpoint /pnr/lab/nurul/PGMT_Code/checkpoints_retrain/xsub_zsl24/epoch_170.pth \
#     --skeleton_dir $LOCAL_SKEL_DIR \
#     --offline_features_dir $LOCAL_PT_DIR

echo "=========================================================="
echo "All High-Speed Evaluations Successfully Completed."
echo "=========================================================="