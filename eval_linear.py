import os
import glob
import time
import argparse

# CACHE FIX: Force Hugging Face and PyTorch Hub to use local folders
os.environ["HF_HOME"] = "/pnr/lab/nurul/PGMT_Code/hf_cache"
os.environ["TORCH_HOME"] = "/pnr/lab/nurul/PGMT_Code/torch_cache"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dataset_ntu_eval import NTURGBD_EvalDataset
from pgmt_architecture import PGMTDualStream

def load_model_weights(model, checkpoint_path):
    """Safely loads weights, stripping DDP/Compile prefixes."""
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('_orig_mod.', '').replace('module.', '')
        clean_state_dict[clean_k] = v
    model.load_state_dict(clean_state_dict, strict=False)
    return model

def extract_features(dataloader, model, device, desc="Extracting"):
    """Passes the dataset through the frozen PGMT encoder ONCE."""
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_gpu = {
                "visual_frames_siglip": batch["visual_frames_siglip"].to(device),
                "visual_frames_dinov2": batch["visual_frames_dinov2"].to(device),
                "kinematic_streams": batch["kinematic_streams"].to(device),
                "spatial_anchors": batch["spatial_anchors"].to(device)
            }
            labels = batch["action_label"]
            B, T = batch["visual_frames_siglip"].shape[:2]
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(batch_gpu)
                
            # Use the semantic graph features for action classification
            graph_feat = outputs["info_nce_graph_sem"]
            
            # Pool across Time dimension (B, T, 1024) -> (B, 1024)
            graph_feat = graph_feat.view(B, T, -1).mean(dim=1)
            # Normalize to match InfoNCE pre-training distribution
            graph_feat = F.normalize(graph_feat.float(), dim=-1, eps=1e-8)
            
            all_features.append(graph_feat.cpu())
            all_labels.extend(labels.tolist())
            
            if (i + 1) % 50 == 0:
                print(f"    - {desc} batch {i+1}/{len(dataloader)}")
                
    return torch.cat(all_features, dim=0), torch.tensor(all_labels)

class LinearProbe(nn.Module):
    """A single supervised layer to evaluate the raw semantic capacity of the latent space."""
    def __init__(self, in_dim=1024, num_classes=120):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        # Initialize weights to zeros for standard linear evaluation protocol
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=str, default="xsub", choices=["xsub", "xview"])
    parser.add_argument("--epochs", type=int, default=100, help="Number of linear probe training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for linear probe")
    args = parser.parse_args()

    print("="*65)
    print(f" PGMT Phase 1: Linear Probe Action Classification ({args.protocol.upper()})")
    print("="*65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_dir = "checkpoints"
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "pgmt_stage1_epoch_*.pth")), 
        key=lambda x: int(os.path.basename(x).split('epoch_')[1].split('.')[0])
    )
    if not checkpoints:
        print("[!] No checkpoints found.")
        return
    checkpoint_file = checkpoints[-1] 
    print(f"[*] Using frozen checkpoint: {checkpoint_file}")

    # 1. Initialize Datasets
    print(f"\n[*] Loading {args.protocol.upper()} Training Set...")
    train_dataset = NTURGBD_EvalDataset(protocol=args.protocol, split="train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"[*] Loading {args.protocol.upper()} Testing Set...")
    test_dataset = NTURGBD_EvalDataset(protocol=args.protocol, split="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 2. Load Frozen PGMT Model
    print("\n[*] Loading Dual-Loss Architecture...")
    model = PGMTDualStream(vis_dim=1024, stage1_pretraining=True).to(device)
    model = load_model_weights(model, checkpoint_file)
    model.eval()

    # 3. Extract Features (Runs the heavy transformer ONCE)
    print(f"\n[*] Extracting Training Features (This will take a few minutes)...")
    train_feats, train_labels = extract_features(train_loader, model, device, "Train")
    
    print(f"\n[*] Extracting Testing Features (This will take a few minutes)...")
    test_feats, test_labels = extract_features(test_loader, model, device, "Test")

    # Clear original model from VRAM as it's no longer needed
    del model
    torch.cuda.empty_cache()

    # 4. Train Linear Probe (Lightning Fast!)
    print(f"\n[*] Memory Bank Size: {train_feats.shape[0]} | Query Size: {test_feats.shape[0]}")
    print(f"[*] Training Supervised Linear Head for {args.epochs} Epochs...")
    
    linear_head = LinearProbe(in_dim=1024, num_classes=120).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(linear_head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create fast feature-only dataloader
    probe_train_dataset = TensorDataset(train_feats, train_labels)
    probe_train_loader = DataLoader(probe_train_dataset, batch_size=1024, shuffle=True)
    
    start_time = time.time()
    for epoch in range(args.epochs):
        linear_head.train()
        total_loss = 0.0
        
        for feats, labels in probe_train_loader:
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = linear_head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"    - Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss/len(probe_train_loader):.4f}")
            
    print(f"[*] Linear Probe Training Complete! (Took {time.time() - start_time:.2f} seconds)")

    # 5. Evaluate Linear Probe
    print("[*] Evaluating on Test Set...")
    linear_head.eval()
    
    test_feats = test_feats.to(device)
    test_labels = test_labels.to(device)
    
    with torch.no_grad():
        logits = linear_head(test_feats)
        _, top1_preds = logits.topk(1, dim=-1)
        
        correct = (top1_preds.squeeze() == test_labels).sum().item()
        accuracy = (correct / test_feats.shape[0]) * 100.0

    print("\n" + "="*65)
    print(f" OFFICIAL {args.protocol.upper()} EVALUATION RESULTS: LINEAR PROBING")
    print("="*65)
    print(f"Test Set: NTU-RGB+D 120 {args.protocol.upper()}")
    print(f"Evaluation Method: Supervised Linear Probing")
    print(f"Frozen Encoder Parameters: ~15M (MS-HyperTR)")
    print(f"\n  Classification Accuracy: {accuracy:.2f}%\n")
    print("="*65)

if __name__ == "__main__":
    main()