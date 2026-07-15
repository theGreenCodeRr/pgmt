import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_ntu import NTURGBD_AlignedDataset
from architecture import PGMTDualStream
import time
import cv2
import multiprocessing

# --- PREVENT CPU THREAD EXPLOSION ---
cv2.setNumThreads(0)

# --- NVIDIA AMPERE (RTX 30-SERIES) OPTIMIZATIONS ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def info_nce_loss(features_a, features_b, temperature=0.07):
    """
    Computes the Contrastive InfoNCE Loss between two modalities.
    """
    features_a = features_a.float()
    features_b = features_b.float()

    features_a = torch.nn.functional.normalize(features_a, dim=-1, eps=1e-5)
    features_b = torch.nn.functional.normalize(features_b, dim=-1, eps=1e-5)
    
    sim_matrix = torch.matmul(features_a, features_b.T) / temperature
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    
    loss_a = nn.CrossEntropyLoss()(sim_matrix, labels)
    loss_b = nn.CrossEntropyLoss()(sim_matrix.T, labels)
    
    return (loss_a + loss_b) / 2.0

def main():
    print("="*50)
    print(" PGMT Phase 1: Stage 1 Alignment Training")
    print("="*50)
    
    batch_size = 32  
    num_epochs = 10
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Device: {device} (TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32})")
    
    if torch.cuda.device_count() > 1:
        print(f"[*] Found {torch.cuda.device_count()} GPUs. DataParallel enabled.")

    print("[*] Initializing Dataset...")
    dataset = NTURGBD_AlignedDataset(data_root=r"E:\datasets\ntu_rgbd")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        prefetch_factor=2, 
        pin_memory=False, 
        persistent_workers=True
    )
    
    if len(dataset) == 0:
        print("[!] No data found. Exiting training loop.")
        return

    print("[*] Initializing Architecture...")
    model = PGMTDualStream(stage1_pretraining=True)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.to(device)
    
    print("[*] Compiling model with torch.compile() for max speed...")
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"[!] Warning: torch.compile() failed or not supported. ({e})")
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    print("\n[*] Commencing Training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0 # Track only successful batches for the average
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            batch_gpu = {
                "visual_frames": batch["visual_frames"].to(device, non_blocking=True),
                "kinematic_windows": batch["kinematic_windows"].to(device, non_blocking=True),
                "spatial_anchors": batch["spatial_anchors"].to(device, non_blocking=True)
            }
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(batch_gpu)
                loss = info_nce_loss(outputs["info_nce_visual"], outputs["info_nce_graph"])
                
            # --- THE NaN INTERCEPTOR ---
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[!] Warning: Dropped Batch [{batch_idx}/{len(dataloader)}] due to corrupted NTU coordinates (NaN/Inf).")
                continue # Skip backprop and prevent it from ruining epoch_loss
                
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            valid_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
        # Safely compute average only using valid batches
        avg_loss = epoch_loss / max(1, valid_batches)
        epoch_time = time.time() - start_time
        print(f"\n>> End of Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Valid Batches: {valid_batches}/{len(dataloader)} | Time: {epoch_time:.2f}s\n")
        
        torch.save(model.state_dict(), f"pgmt_stage1_epoch{epoch+1}.pth")
        print(f"[*] Checkpoint saved: pgmt_stage1_epoch{epoch+1}.pth\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()