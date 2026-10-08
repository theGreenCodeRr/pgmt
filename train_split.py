import os
import sys
import gc
import glob
import argparse
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_split import NTURGBD_SplitDataset
from pgmt_split_architecture import PGMTSplitStream

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_trainable_state_dict(model):
    """Safely extracts requires_grad=True weights bypassing detached state_dict flags."""
    trainable_names = [k for k, v in model.named_parameters() if v.requires_grad]
    full_state_dict = model.state_dict()
    return {k: full_state_dict[k] for k in trainable_names}

def infonce_loss(visual_features, graph_features, logit_scale):
    logits = logit_scale * visual_features @ graph_features.T
    labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

def train_chunk(model, dataloader, optimizer, device):
    model.train()
    total_geo = 0.0
    total_sem = 0.0
    
    # HPC-Safe tqdm: Prevents SLURM log file spam by updating only every 30 seconds
    pbar = tqdm(dataloader, desc="  Batch Progress", leave=False, ascii=True, mininterval=30.0, file=sys.stdout)
    
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
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_dict)
            loss_geo = infonce_loss(outputs["info_nce_visual_geo"], outputs["info_nce_graph_geo"], outputs["logit_scale_geo"])
            loss_sem = infonce_loss(outputs["info_nce_visual_sem"], outputs["info_nce_graph_sem"], outputs["logit_scale_sem"])
            loss = loss_geo + loss_sem
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_geo += loss_geo.item()
        total_sem += loss_sem.item()
        
        # Explicitly display separated dual losses for graphing
        pbar.set_postfix(GeoLoss=f"{loss_geo.item():.4f}", SemLoss=f"{loss_sem.item():.4f}", refresh=False)
        
    return total_geo / len(dataloader), total_sem / len(dataloader)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=str, default="xsub", choices=["xsub", "xset"])
    parser.add_argument("--zsl_classes", type=int, default=0, choices=[0, 10, 24])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    
    run_name = f"{args.protocol}_zsl{args.zsl_classes}"
    checkpoint_dir = f"/pnr/lab/nurul/PGMT_Code/checkpoints_retrain/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info(f"[*] Starting Split-Train: {run_name.upper()} on {device} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    
    model = PGMTSplitStream(vis_dim=1024, stage1_pretraining=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    # Read instantly from the hardcoded surviving NVMe path
    slurm_tmp = os.environ.get("SLURM_TMPDIR", "/tmp")
    chunk_files = sorted(glob.glob(os.path.join(slurm_tmp, "ntu_features_chunk_*.pt")))
    
    if not chunk_files:
        logger.error(f"[!] No feature chunks found in {slurm_tmp}. The hijack failed.")
        sys.exit(1)
    
    for epoch in range(args.epochs):
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
                
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            
            chunk_geo, chunk_sem = train_chunk(model, dataloader, optimizer, device)
            epoch_geo += chunk_geo
            epoch_sem += chunk_sem
            active_chunks += 1
            
            del chunk_data, dataset, dataloader
            gc.collect()
            
        scheduler.step()
        avg_geo = epoch_geo / max(1, active_chunks)
        avg_sem = epoch_sem / max(1, active_chunks)
        total_avg = avg_geo + avg_sem
        
        # Log all three metrics required for your graphing
        logger.info(f"Epoch {epoch+1} Complete | GeoLoss: {avg_geo:.4f} | SemLoss: {avg_sem:.4f} | Total: {total_avg:.4f}")
        
        # Save fixed ~60MB checkpoints
        trainable_weights = get_trainable_state_dict(model)
        save_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        torch.save(trainable_weights, save_path)
        logger.info(f"[*] Saved optimized Phase 1 weights to {save_path}")

if __name__ == "__main__":
    main()
