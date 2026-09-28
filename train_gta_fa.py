import os
import sys  # Added for SLURM stdout routing
# Force Hugging Face to use a local cache folder to bypass cluster permission errors
os.environ["HF_HOME"] = "/pnr/lab/nurul/PGMT_Code/hf_cache"
# Force PyTorch Hub (for DINOv2) to use a local cache folder!
os.environ["TORCH_HOME"] = "/pnr/lab/nurul/PGMT_Code/torch_cache"

# Ensure logs directory exists before configuring the logger
os.makedirs("logs", exist_ok=True)
os.makedirs("torch_cache", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("hf_cache", exist_ok=True)

import glob
import argparse
import logging
import contextlib
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Import our custom PGMT modules
from dataset_ntu import NTURGBD_AlignedDataset
from pgmt_architecture import PGMTDualStream

# Configure logging to write to BOTH the terminal and a dedicated file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/pgmt_python_app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_distributed():
    """Initializes PyTorch DDP via NCCL backend."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

# =================================================================
# GLOBAL CONTRASTIVE FIX: Differentiable All-Gather Layer
# =================================================================
class GatherLayer(torch.autograd.Function):
    """Gathers tensors from all GPUs while preserving backpropagation gradients."""
    @staticmethod
    def forward(ctx, x):
        output = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def gather_features(features):
    if dist.is_initialized():
        gathered = GatherLayer.apply(features)
        return torch.cat(gathered, dim=0)
    return features
# =================================================================

def infonce_loss(features_a, features_b, logit_scale=None, temperature=0.07):
    """
    High-Capacity InfoNCE Loss using global GPU aggregation.
    """
    # Normalize local features
    features_a = F.normalize(features_a.float(), dim=-1, eps=1e-8)
    features_b = F.normalize(features_b.float(), dim=-1, eps=1e-8)
    
    # Broadcast features across all GPUs to create massive negative pool
    global_features_a = gather_features(features_a)
    global_features_b = gather_features(features_b)
    
    if logit_scale is not None:
        logits = torch.matmul(features_a, global_features_b.T) * logit_scale.float()
    else:
        logits = torch.matmul(features_a, global_features_b.T) / temperature
    
    # Calculate global target labels based on GPU rank
    batch_size = features_a.size(0)
    rank = dist.get_rank() if dist.is_initialized() else 0
    labels = torch.arange(batch_size, device=features_a.device) + (rank * batch_size)
    
    loss_a = F.cross_entropy(logits, labels)
    
    # Symmetric loss calculation
    if logit_scale is not None:
        logits_T = torch.matmul(features_b, global_features_a.T) * logit_scale.float()
    else:
        logits_T = torch.matmul(features_b, global_features_a.T) / temperature
        
    loss_b = F.cross_entropy(logits_T, labels)
    
    return (loss_a + loss_b) / 2.0


def get_trainable_state_dict(model):
    """
    Extracts only the weights that have requires_grad=True.
    This strips out the frozen SigLIP/DINOv2 backbones, shrinking checkpoint sizes dramatically.
    """
    state_dict = model.state_dict()
    trainable_keys = [k for k, p in model.named_parameters() if p.requires_grad]
    return {k: state_dict[k] for k in trainable_keys}


def train_epoch(model, dataloader, optimizer, accumulation_steps, local_rank):
    model.train()
    total_loss = 0.0
    
    # SLURM TQDM FIX: Drastically reduces terminal/log spam
    pbar = tqdm(
        dataloader, 
        disable=(local_rank != 0), 
        desc="Training",
        file=sys.stdout,   # SLURM handles stdout much cleaner than stderr
        mininterval=30.0,  # ONLY print an update to the log file every 30 seconds
        ascii=True         # Prevent ugly Unicode character corruption in .log files
    )
    
    for batch_idx, batch in enumerate(pbar):
        input_dict = {
            "visual_frames_siglip": batch["visual_frames_siglip"].to(local_rank, non_blocking=True),
            "visual_frames_dinov2": batch["visual_frames_dinov2"].to(local_rank, non_blocking=True),
            "kinematic_streams": batch["kinematic_streams"].to(local_rank, non_blocking=True),
            "spatial_anchors": batch["spatial_anchors"].to(local_rank, non_blocking=True)
        }

        is_accumulating = (batch_idx + 1) % accumulation_steps != 0 and (batch_idx + 1) != len(dataloader)
        context_manager = model.no_sync() if is_accumulating else contextlib.nullcontext()

        with context_manager:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_dict)
                
                # -----------------------------------------------------
                # PATHWAY A: Geometric InfoNCE (Fused SigLIP + DINOv2)
                # -----------------------------------------------------
                v_target_geo = outputs["info_nce_visual_geo"]
                g_target_geo = outputs["info_nce_graph_geo"]
                scale_geo = outputs.get("logit_scale_geo", None)
                loss_geo = infonce_loss(v_target_geo, g_target_geo, logit_scale=scale_geo)
                
                # -----------------------------------------------------
                # PATHWAY B: Semantic InfoNCE (Pure Text-Aligned SigLIP)
                # -----------------------------------------------------
                v_target_sem = outputs["info_nce_visual_sem"]
                g_target_sem = outputs["info_nce_graph_sem"]
                scale_sem = outputs.get("logit_scale_sem", None)
                loss_sem = infonce_loss(v_target_sem, g_target_sem, logit_scale=scale_sem)
                
                # Combine both paths for the master dual-loss
                loss = (loss_geo + loss_sem) / accumulation_steps 

            loss.backward()

        if not is_accumulating:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        real_loss = loss.item() * accumulation_steps
        total_loss += real_loss
        
        if local_rank == 0:
            #  refresh=False FIX: Updates the loss numbers without forcing a new line print
            pbar.set_postfix(
                GeoLoss=f"{loss_geo.item():.3f}", 
                SemLoss=f"{loss_sem.item():.3f}",
                refresh=False
            )
            
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--accumulation_steps", type=int, default=4) 
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    local_rank = setup_distributed()
    if local_rank == 0: logger.info("DDP Initialized via NCCL. Starting PGMT Stage 1 Training.")

    dataset = NTURGBD_AlignedDataset(data_root="/pnr/lab/nurul/datasets/ntu_rgbd")
    sampler = DistributedSampler(dataset)
    
    # REDUCED WORKERS: Drops background processes to prevent System RAM OOM
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, 
        num_workers=2, pin_memory=True, drop_last=True
    )

    if local_rank == 0:
        logger.info("[*] Rank 0: Initializing model and checking HF/Torch weights...")
        model = PGMTDualStream(vis_dim=1024, stage1_pretraining=True).to(local_rank)
        
    dist.barrier() 
    
    if local_rank != 0:
        logger.info(f"[*] Rank {local_rank}: Loading cached HF/Torch weights...")
        model = PGMTDualStream(vis_dim=1024, stage1_pretraining=True).to(local_rank)

    if local_rank == 0: logger.info("Skipping compilation to bypass HuggingFace Dynamo bug...")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    start_epoch = 0
    checkpoint_dir = "checkpoints"
    
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "pgmt_stage1_epoch_*.pth")), 
        key=lambda x: int(os.path.basename(x).split('epoch_')[1].split('.')[0])
    )
    
    if checkpoints:
        latest_ckpt = checkpoints[-1]
        start_epoch = int(os.path.basename(latest_ckpt).split('epoch_')[1].split('.')[0])
        
        if local_rank == 0: 
            logger.info(f"[*] Found existing checkpoint: {latest_ckpt}")
            logger.info(f"[*] Resuming training from Epoch {start_epoch + 1}...")
            
        state_dict = torch.load(latest_ckpt, map_location=f"cuda:{local_rank}", weights_only=True)
        # THE STRICT=FALSE FIX: Allow DINOv2 and SigLIP to stay frozen while MS-HyperTR updates
        model.module.load_state_dict(state_dict, strict=False)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch) 
        if local_rank == 0: logger.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        avg_loss = train_epoch(model, dataloader, optimizer, args.accumulation_steps, local_rank)
        
        if local_rank == 0:
            logger.info(f"Epoch {epoch+1} Complete. Avg Dual-Loss: {avg_loss:.4f}")
            
            # Save only trainable parameters to keep files at ~20MB
            trainable_weights = get_trainable_state_dict(model.module)
            save_path = os.path.join(checkpoint_dir, f"pgmt_stage1_epoch_{epoch+1}.pth")
            torch.save(trainable_weights, save_path)
            logger.info(f"[*] Saved optimized Phase 1 weights to {save_path}")

    cleanup_distributed()

if __name__ == "__main__":
    main()