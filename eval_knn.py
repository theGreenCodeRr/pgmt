import os
# CACHE FIX: Force Hugging Face and PyTorch Hub to use local folders
os.environ["HF_HOME"] = "/pnr/lab/nurul/PGMT_Code/hf_cache"
os.environ["TORCH_HOME"] = "/pnr/lab/nurul/PGMT_Code/torch_cache"

import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_ntu_eval import NTURGBD_EvalDataset
from pgmt_architecture import PGMTDualStream
import argparse

def load_model_weights(model, checkpoint_path):
    """Safely loads weights, stripping DDP/Compile prefixes."""
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('_orig_mod.', '').replace('module.', '')
        clean_state_dict[clean_k] = v
    # strict=False allows us to load the tiny 20MB file while keeping Google/Meta backbones frozen
    model.load_state_dict(clean_state_dict, strict=False)
    return model

def extract_features(dataloader, model, device, desc="Extracting"):
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
            # This space is directly aligned with the pure text representation
            graph_feat = outputs["info_nce_graph_sem"]
            
            # Pool across Time dimension (B, T, 1024) -> (B, 1024)
            graph_feat = graph_feat.view(B, T, -1).mean(dim=1)
            graph_feat = F.normalize(graph_feat.float(), dim=-1, eps=1e-8)
            
            all_features.append(graph_feat.cpu())
            all_labels.extend(labels.tolist())
            
            if (i + 1) % 50 == 0:
                print(f"    - {desc} batch {i+1}/{len(dataloader)}")
                
    return torch.cat(all_features, dim=0), torch.tensor(all_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=str, default="xsub", choices=["xsub", "xview"], 
                        help="Choose 'xsub' (Cross-Subject) or 'xview' (Cross-View)")
    args = parser.parse_args()

    print("="*65)
    print(f" PGMT Phase 1: k-NN Action Classification ({args.protocol.upper()})")
    print("="*65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print(f"[!] Cannot find '{checkpoint_dir}/' folder. Wait for the training epoch to finish!")
        return

    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "pgmt_stage1_epoch_*.pth")), 
        key=lambda x: int(os.path.basename(x).split('epoch_')[1].split('.')[0])
    )
    if not checkpoints:
        print("[!] No checkpoints found.")
        return
    checkpoint_file = checkpoints[-1] 
    print(f"[*] Using checkpoint: {checkpoint_file}")

    # 1. Initialize Datasets (Train for Memory Bank, Test for Queries)
    print(f"\n[*] Loading {args.protocol.upper()} Training Set (Memory Bank)...")
    train_dataset = NTURGBD_EvalDataset(protocol=args.protocol, split="train")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    print(f"\n[*] Loading {args.protocol.upper()} Testing Set (Queries)...")
    test_dataset = NTURGBD_EvalDataset(protocol=args.protocol, split="test")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 2. Load Model
    print("\n[*] Loading Dual-Loss Architecture...")
    model = PGMTDualStream(vis_dim=1024, stage1_pretraining=True).to(device)
    model = load_model_weights(model, checkpoint_file)
    model.eval()

    # 3. Extract Features
    print("\n[*] Extracting Training Features (This will take a few minutes)...")
    train_feats, train_labels = extract_features(train_loader, model, device, "Train")
    
    print("\n[*] Extracting Testing Features...")
    test_feats, test_labels = extract_features(test_loader, model, device, "Test")

    # Move to GPU for fast matrix math
    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    test_feats = test_feats.to(device)
    test_labels = test_labels.to(device)

    print(f"\n[*] Memory Bank Size: {train_feats.shape[0]} | Query Size: {test_feats.shape[0]}")
    print("[*] Running 1-NN Classification...")

    # Memory-safe chunked k-NN
    chunk_size = 2048
    correct = 0
    total = test_feats.shape[0]
    
    for i in range(0, total, chunk_size):
        end = min(i + chunk_size, total)
        query_chunk = test_feats[i:end]
        
        # Cosine similarity between Queries and Memory Bank
        sim_matrix = torch.matmul(query_chunk, train_feats.T)
        
        # Get index of highest similarity (Top-1 Nearest Neighbor)
        _, top1_idx = sim_matrix.max(dim=-1)
        
        # Get the action label of that nearest neighbor
        pred_labels = train_labels[top1_idx]
        true_labels = test_labels[i:end]
        
        correct += (pred_labels == true_labels).sum().item()

    accuracy = (correct / total) * 100.0

    print("\n" + "="*65)
    print(f" OFFICIAL {args.protocol.upper()} EVALUATION RESULTS: ACTION CLASSIFICATION")
    print("="*65)
    print(f"Test Set: NTU-RGB+D 120 {args.protocol.upper()}")
    print(f"Latent Space: Semantic (Pure SigLIP Alignment)")
    print(f"Evaluation Method: 1-Nearest Neighbor (k-NN) on Graph Embeddings")
    print(f"\n  Classification Accuracy: {accuracy:.2f}%\n")
    print("="*65)

if __name__ == "__main__":
    main()