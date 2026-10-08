#!/bin/bash
#SBATCH --job-name=PGMT_SplitTrain
#SBATCH --partition=d
#SBATCH --nodelist=oniwaka
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=65G
#SBATCH --output=logs/pgmt_split_training_%j.log

# Abort immediately if any command fails
set -e

# Explicitly define and create the local NVMe scratch folder
#export SLURM_TMPDIR="/tmp/$SLURM_JOB_ID"
#mkdir -p $SLURM_TMPDIR

export SLURM_TMPDIR="/tmp/13357"

# STRICT SAFETY GUARDRAIL: Wipe local NVMe storage instantly on exit, interrupt, or terminate
#trap 'echo "[*] Job terminated. Wiping local NVMe storage..."; rm -rf $SLURM_TMPDIR; echo "[*] Cleanup complete."' EXIT INT TERM

export OMP_NUM_THREADS=1
export HF_HOME="/pnr/lab/nurul/PGMT_Code/hf_cache"
export TORCH_HOME="/pnr/lab/nurul/PGMT_Code/torch_cache"

#echo "=========================================================="
#echo "Starting Tier 3 Multi-Split Retraining Pipeline"
#echo "Target Node: $SLURMD_NODENAME"
#echo "=========================================================="

# 1. Stage the Visual Features
#echo "[*] Copying 12 feature chunks from NAS to Local NVMe ($SLURM_TMPDIR)..."
#cp /pnr/lab/nurul/PGMT_Code/offline_features/ntu_features_chunk_*.pt $SLURM_TMPDIR/

# 2. Stage the Kinematics via Single Archive
#echo "[*] Copying single skeleton archive to local NVMe..."
#cp /pnr/lab/nurul/datasets/ntu_rgbd/ntu_skeletons.tar $SLURM_TMPDIR/

#echo "[*] Extracting skeletons on local NVMe..."
#tar -xf $SLURM_TMPDIR/ntu_skeletons.tar -C $SLURM_TMPDIR/
#rm -f $SLURM_TMPDIR/ntu_skeletons.tar
#echo "[*] Data staging complete. Fast local I/O initialized."

echo "[*] Data hijack complete. Fast local I/O initialized."

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate /pnr/lab/nurul/PGMT_Code/env

# 3. Execute Fast Training 
echo "[*] Launching Protocol: X-Sub | ZSL: 0"
python train_split.py --protocol xsub --zsl_classes 0 --batch_size 256

echo "[*] Launching Protocol: X-Set | ZSL: 0"
python train_split.py --protocol xset --zsl_classes 0 --batch_size 256

echo "[*] Launching Protocol: X-Sub | ZSL: 10"
python train_split.py --protocol xsub --zsl_classes 10 --batch_size 256

echo "[*] Launching Protocol: X-Sub | ZSL: 24"
python train_split.py --protocol xsub --zsl_classes 24 --batch_size 256

echo "=========================================================="
echo "All 4 Retraining Protocols Successfully Completed."
echo "=========================================================="
