import os
import sys
import gc
import glob
import math
import argparse
import logging
import contextlib

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset_split import NTURGBD_SplitDataset
from pgmt_split_architecture import PGMTSplitStream

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

class GatherLayer(torch.autograd.Function):
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

def infonce_loss(features_a, features_b, logit_scale):
    global_a = gather_features(features_a)
    global_b = gather_features(features_b)
    
    logits = torch.matmul(features_a, global_b.T) * logit_scale.float()
    
    batch_size = features_a.size(0)
    rank = dist.get_rank() if dist.is_initialized() else 0
    labels = torch.arange(batch_size, device=features_a.device) + (rank * batch_size)
    
    loss_a = F.cross_entropy(logits, labels)
    logits_T = torch.matmul(features_b, global_a.T) * logit_scale.float()
    loss_b = F.cross_entropy(logits_T, labels)
    
    return (loss_a + loss_b) / 2.0

def get_trainable_state_dict(model):
    trainable_names = [k for k, v in model.named_parameters() if v.requires_grad]
    full_state_dict = model.state_dict()
    return {k: full_state_dict[k] for k in trainable_names}

def get_sem_weight(epoch):
    if epoch < 30:
        return 0.0
    elif epoch >= 60:
        return 1.0
    else:
        return (epoch - 30) / 30.0

def train_chunk(model, dataloader, optimizer, device, epoch, local_rank):
    model.train()
    total_geo = 0.0
    total_sem = 0.0
    
    sem_weight = get_sem_weight(epoch)
    pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1}", disable=(local_rank != 0), leave=False, ascii=True, mininterval=30.0, file=sys.stdout)
    
    for batch in pbar:
        B, T = batch["geo_fused"].shape[:2]
        dummy_anchors = (torch.ones(B, T, 2, 2) * 0.5).to(device)
        
        input_dict = {
            "geo_fused": batch["geo_fused"].to(device, non_blocking=True),
            "sem_global": batch["sem_global"].to(device, non_blocking=True),
            "kinematic_streams": batch["kinematic_streams"].to(device, non_blocking=True),
            "spatial_anchors": dummy_anchors
        }
        
        optimizer.zero_grad()
        
        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_dict)
                loss_geo = infonce_loss(outputs["info_nce_visual_geo"], outputs["info_nce_graph_geo"], outputs["logit_scale_geo"])
                loss_sem = infonce_loss(outputs["info_nce_visual_sem"], outputs["info_nce_graph_sem"], outputs["logit_scale_sem"])
                
                loss = loss_geo + (loss_sem * sem_weight)
                
            loss.backward()
            
            # Type-safe gradient norm calculation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # --- HEALTH CHECK: GRADIENT FLOW MONITOR ---
            if torch.isnan(grad_norm) or grad_norm.item() < 1e-6:
                logger.error(f"[!] FATAL: Network Collapse Detected on Rank {local_rank} (Grad Norm: {grad_norm.item():.6f}). The Transformer exploded and was zeroed out. Killing job.")
                cleanup_distributed()
                sys.exit(1)
                
            optimizer.step()
            
        except RuntimeError as e:
            # Gracefully catch the Sniffer's explosion and tear down the DDP cluster
            if "FATAL EXPLOSION" in str(e):
                logger.error(f"\n[!] DIAGNOSTIC TRIGGERED on Rank {local_rank}:\n{str(e)}\n")
                cleanup_distributed()
                sys.exit(1)
            else:
                raise e
        
        total_geo += loss_geo.item()
        total_sem += loss_sem.item()
        
        if local_rank == 0:
            pbar.set_postfix(GeoLoss=f"{loss_geo.item():.4f}", SemLoss=f"{loss_sem.item():.4f}", SemWt=f"{sem_weight:.2f}", refresh=False)
            
    return total_geo / len(dataloader), total_sem / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=str, default="xsub", choices=["xsub", "xset"])
    parser.add_argument("--zsl_classes", type=int, default=10, choices=[0, 10, 24])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_dir", type=str, default="/pnr/lab/nurul/PGMT_Code/checkpoints_retrain")
    args = parser.parse_args()
    
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    run_name = f"{args.protocol}_zsl{args.zsl_classes}"
    checkpoint_dir = os.path.join(args.save_dir, run_name)
    
    if local_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"[*] DDP Split-Train: {run_name.upper()} | Epochs: {args.epochs} | Local Batch: {args.batch_size}")
        logger.info(f"[*] Checkpoint Directory: {checkpoint_dir}")
    
    model = PGMTSplitStream(vis_dim=1024, stage1_pretraining=True).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # --- ACTIVATE THE NUMERICAL SNIFFER ---
    if hasattr(model.module, 'register_nan_hooks'):
        model.module.register_nan_hooks()
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    chunk_files = sorted(glob.glob("/tmp/feature_cache/ntu_features_chunk_*.pt"))
    if not chunk_files:
        if local_rank == 0:
            logger.error("[!] No feature chunks found in /tmp/feature_cache. Staging failed.")
        cleanup_distributed()
        sys.exit(1)

    # --- PERFECT AUTO-RESUME LOGIC ---
    start_epoch = 0
    existing_ckpts = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth"))
    if existing_ckpts:
        start_epoch = max([int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_ckpts])
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{start_epoch}.pth")
        
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.module.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        if local_rank == 0:
            logger.info(f"[*] Found checkpoint {ckpt_path}. Resuming perfectly from Epoch {start_epoch + 1}...")

    for epoch in range(start_epoch, args.epochs):
        if local_rank == 0:
            logger.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
            
        epoch_geo = 0.0
        epoch_sem = 0.0
        active_chunks = 0
        
        for chunk_file in chunk_files:
            chunk_data = torch.load(chunk_file, map_location="cpu", weights_only=True)
            
            dataset = NTURGBD_SplitDataset(
                chunk_data=chunk_data, 
                skeleton_dir=None, 
                protocol=args.protocol, 
                split="train", 
                zsl_classes=args.zsl_classes
            )
            
            if len(dataset) == 0:
                del chunk_data, dataset
                gc.collect()
                continue
                
            sampler = DistributedSampler(dataset, shuffle=True)
            sampler.set_epoch(epoch)
            
            dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
            
            chunk_geo, chunk_sem = train_chunk(model, dataloader, optimizer, device, epoch, local_rank)
            epoch_geo += chunk_geo
            epoch_sem += chunk_sem
            active_chunks += 1
            
            del chunk_data, dataset, dataloader, sampler
            gc.collect()
            
        scheduler.step()
        
        if local_rank == 0:
            avg_geo = epoch_geo / max(1, active_chunks)
            avg_sem = epoch_sem / max(1, active_chunks)
            total_avg = avg_geo + avg_sem
            logger.info(f"Epoch {epoch+1} Complete | GeoLoss: {avg_geo:.4f} | SemLoss: {avg_sem:.4f} | Total: {total_avg:.4f}")
            
            checkpoint_state = {
                'model': get_trainable_state_dict(model.module),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            save_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            torch.save(checkpoint_state, save_path)
            logger.info(f"[*] Saved perfect checkpoint to {save_path}")
            
    cleanup_distributed()

if __name__ == "__main__":
    main()