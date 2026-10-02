import os
import sys

# Force Hugging Face & PyTorch Hub to use local cache folders to bypass cluster permission blocks
os.environ["HF_HOME"] = "/pnr/lab/nurul/PGMT_Code/hf_cache"
os.environ["TORCH_HOME"] = "/pnr/lab/nurul/PGMT_Code/torch_cache"

os.makedirs("logs", exist_ok=True)
os.makedirs("torch_cache", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
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

# Configure dual logging (Terminal + File)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/pgmt_python_app.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def setup_distributed():
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
# LOSS FUNCTION
# =================================================================
def infonce_loss(features_a, features_b, logit_scale=None, temperature=0.07):
    features_a = F.normalize(features_a.float(), dim=-1, eps=1e-8)
    features_b = F.normalize(features_b.float(), dim=-1, eps=1e-8)
    
    global_features_a = gather_features(features_a)
    global_features_b = gather_features(features_b)
    
    if logit_scale is not None:
        logits = torch.matmul(features_a, global_features_b.T) * logit_scale.float()
        logits_T = torch.matmul(features_b, global_features_a.T) * logit_scale.float()
    else:
        logits = torch.matmul(features_a, global_features_b.T) / temperature
        logits_T = torch.matmul(features_b, global_features_a.T) / temperature
    
    batch_size = features_a.size(0)
    rank = dist.get_rank() if dist.is_initialized() else 0
    labels = torch.arange(batch_size, device=features_a.device) + (rank * batch_size)
    
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits_T, labels)
    
    return (loss_a + loss_b) / 2.0

def get_trainable_state_dict(model):
    state_dict = model.state_dict()
    trainable_keys = [k for k, p in model.named_parameters() if p.requires_grad]
    return {k: state_dict[k] for k in trainable_keys}

# =================================================================
# TRAINING LOOP
# =================================================================
def train_epoch(model, dataloader, optimizer, scheduler, accumulation_steps, local_rank):
    model.train()
    total_loss = 0.0
    
    # SLURM-safe tqdm to prevent massive log files
    pbar = tqdm(
        dataloader, 
        disable=(local_rank != 0), 
        desc="Training",
        file=sys.stdout,   
        mininterval=30.0,  
        ascii=True         
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
                
                # Geometric Grounding (Fused)
                v_target_geo = outputs["info_nce_visual_geo"]
                g_target_geo = outputs["info_nce_graph_geo"]
                scale_geo = outputs.get("logit_scale_geo", None)
                loss_geo = infonce_loss(v_target_geo, g_target_geo, logit_scale=scale_geo)
                
                # Semantic Grounding (Global)
                v_target_sem = outputs["info_nce_visual_sem"]
                g_target_sem = outputs["info_nce_graph_sem"]
                scale_sem = outputs.get("logit_scale_sem", None)
                loss_sem = infonce_loss(v_target_sem, g_target_sem, logit_scale=scale_sem)
                
                loss = (loss_geo + loss_sem) / accumulation_steps 

            loss.backward()

        if not is_accumulating:
            # Prevent gradient explosion from corrupted samples
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        real_loss = loss.item() * accumulation_steps
        total_loss += real_loss
        
        if local_rank == 0:
            pbar.set_postfix(
                GeoLoss=f"{loss_geo.item():.3f}", 
                SemLoss=f"{loss_sem.item():.3f}",
                refresh=False
            )
            
    # Step the learning rate schedule at the end of the epoch
    scheduler.step()
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--accumulation_steps", type=int, default=4) 
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--data_root", type=str, default="/pnr/lab/nurul/datasets/ntu_rgbd") # NEW LINE
    args = parser.parse_args()

    local_rank = setup_distributed()
    if local_rank == 0: logger.info("DDP Initialized via NCCL. Starting PGMT Stage 1 Training.")

    dataset = NTURGBD_AlignedDataset(data_root=args.data_root) # UPDATED LINE
    sampler = DistributedSampler(dataset)
    
    # Strictly capped workers/prefetch to prevent OOM on 12GB free RAM
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, 
        num_workers=12, 
        prefetch_factor=4, 
        pin_memory=True, 
        drop_last=True
    )

    if local_rank == 0:
        logger.info("[*] Rank 0: Initializing model...")
        model = PGMTDualStream(vis_dim=1024, stage1_pretraining=True).to(local_rank)
        
    dist.barrier() 
    
    if local_rank != 0:
        model = PGMTDualStream(vis_dim=1024, stage1_pretraining=True).to(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Cosine Annealing to stabilize final 20 epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

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
            
        checkpoint_data = torch.load(latest_ckpt, map_location=f"cuda:{local_rank}", weights_only=False)
        
        # Backward compatibility for resuming from legacy checkpoints vs new state dicts
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            model.module.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        else:
            model.module.load_state_dict(checkpoint_data, strict=False)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch) 
        if local_rank == 0: logger.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, args.accumulation_steps, local_rank)
        
        if local_rank == 0:
            logger.info(f"Epoch {epoch+1} Complete. Avg Dual-Loss: {avg_loss:.4f}")
            
            # Save comprehensive state dictionary
            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state_dict': get_trainable_state_dict(model.module),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss
            }
            
            save_path = os.path.join(checkpoint_dir, f"pgmt_stage1_epoch_{epoch+1}.pth")
            torch.save(checkpoint_state, save_path)
            logger.info(f"[*] Saved Phase 1 State Dictionary to {save_path}")

    cleanup_distributed()

if __name__ == "__main__":
    main()