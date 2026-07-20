import os
import argparse
import logging
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_distributed():
    """Initializes PyTorch DDP via NCCL backend."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def infonce_loss(features_a, features_b, temperature=0.07):
    """
    High-Capacity InfoNCE Loss.
    Calculates an N x N similarity matrix to penalize hard negatives.
    """
    # L2 Normalize features to the unit hypersphere
    features_a = F.normalize(features_a, dim=-1)
    features_b = F.normalize(features_b, dim=-1)
    
    # Calculate similarity matrix (Batch*T x Batch*T)
    logits = torch.matmul(features_a, features_b.T) / temperature
    
    # Labels are the diagonal (each token matches with itself across modalities)
    batch_size = features_a.size(0)
    labels = torch.arange(batch_size, device=features_a.device)
    
    # Symmetric cross-entropy loss (Vision->Graph and Graph->Vision)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    
    return (loss_a + loss_b) / 2.0

def train_epoch(model, dataloader, optimizer, accumulation_steps, local_rank):
    model.train()
    total_loss = 0.0
    
    # Only show progress bar on the primary GPU (Rank 0)
    pbar = tqdm(dataloader, disable=(local_rank != 0), desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        # 1. Move batch to the correct GPU
        input_dict = {
            "visual_frames_siglip": batch["visual_frames_siglip"].to(local_rank, non_blocking=True),
            "visual_frames_dinov2": batch["visual_frames_dinov2"].to(local_rank, non_blocking=True),
            "kinematic_streams": batch["kinematic_streams"].to(local_rank, non_blocking=True),
            "spatial_anchors": batch["spatial_anchors"].to(local_rank, non_blocking=True)
        }

        # Check if we are accumulating or syncing on this step
        is_accumulating = (batch_idx + 1) % accumulation_steps != 0 and (batch_idx + 1) != len(dataloader)

        # 2. Forward & Backward Pass (with or without DDP sync)
        # Using model.no_sync() prevents the slow PCIe communication on oniwaka during accumulation!
        context_manager = model.no_sync() if is_accumulating else torch.cuda.amp.autocast(enabled=False) # Dummy context if syncing

        with context_manager:
            # Native Hopper bfloat16 for immense speed and stability
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_dict)
                
                v_target = outputs["info_nce_visual"]
                g_target = outputs["info_nce_graph"]
                
                loss = infonce_loss(v_target, g_target)
                loss = loss / accumulation_steps # Scale loss

            loss.backward()

        # 3. Optimizer Step (Only when accumulation is finished)
        if not is_accumulating:
            # Gradient clipping to ensure stability early in training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        
        if local_rank == 0:
            pbar.set_postfix({'InfoNCE': f"{loss.item() * accumulation_steps:.4f}"})
            
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64) # Per GPU limit for 80GB VRAM
    parser.add_argument("--accumulation_steps", type=int, default=4) # Virtual batch = 64 * 2 GPUs * 4 steps
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    # 1. Initialize Distributed Data Parallel
    local_rank = setup_distributed()
    if local_rank == 0: logger.info("DDP Initialized via NCCL. Starting PGMT Stage 1 Training.")

    # 2. Load Dataset (Directly from NAS)
    dataset = NTURGBD_AlignedDataset(data_root="/pnr/lab/nurul/datasets/ntu_rgbd")
    
    # 3. Distributed Sampler & Loader
    # Limits num_workers to 4 per GPU (8 total) to protect oniwaka's 94GB system RAM
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )

    # 4. Build Model & Move to DDP
    model = PGMTDualStream(vis_dim=1024, stage1_pretraining=True).to(local_rank)
    
    # PyTorch 2.x Compilation for H100 FlashAttention integration
    if local_rank == 0: logger.info("Compiling model for Hopper Architecture (This takes ~1 min)...")
    model = torch.compile(model)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 5. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 6. Training Loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch) # Crucial for data shuffling in DDP
        
        if local_rank == 0: logger.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        avg_loss = train_epoch(model, dataloader, optimizer, args.accumulation_steps, local_rank)
        
        if local_rank == 0:
            logger.info(f"Epoch {epoch+1} Complete. Avg InfoNCE Loss: {avg_loss:.4f}")
            # Save Checkpoint
            torch.save(model.module.state_dict(), f"pgmt_stage1_epoch_{epoch+1}.pth")

    cleanup_distributed()

if __name__ == "__main__":
    main()