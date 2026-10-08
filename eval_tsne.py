import os
import argparse
import csv
# CACHE FIX: Force HF, PyTorch, AND Matplotlib to use local folders
os.environ["HF_HOME"] = "/pnr/lab/nurul/PGMT_Code/hf_cache"
os.environ["TORCH_HOME"] = "/pnr/lab/nurul/PGMT_Code/torch_cache"
os.environ["MPLCONFIGDIR"] = "/pnr/lab/nurul/PGMT_Code/matplotlib_cache" 

os.makedirs("outputs", exist_ok=True)
os.makedirs("matplotlib_cache", exist_ok=True)

import glob
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from dataset_ntu_eval import NTURGBD_EvalDataset
from pgmt_architecture import PGMTDualStream

# 10 Kinematically Orthogonal Classes for Clean Unsupervised Clustering
TARGET_CLASSES = {
    8: "standing up",
    23: "kicking something",
    26: "jump up",
    37: "salute",
    42: "falling",
    54: "hugging",
    60: "walking apart",
    83: "cross arms",
    90: "butt kicks",
    102: "yawn" 
}

def load_model_weights(model, checkpoint_path):
    """Safely loads weights, stripping DDP/Compile prefixes."""
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('_orig_mod.', '').replace('module.', '')
        clean_state_dict[clean_k] = v
    model.load_state_dict(clean_state_dict, strict=False)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/pnr/lab/nurul/datasets/ntu_rgbd")
    parser.add_argument("--protocol", type=str, default="all", choices=["all", "xsub", "xset", "xview"],
                        help="Partition protocol. Use 'all' for full uncurated dataset.")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "test"],
                        help="Data split. Use 'all' for full uncurated dataset.")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Explicit path to checkpoint file")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print("="*65)
    print(" PGMT Phase 1: t-SNE Semantic Latent Space Export")
    print(f" Protocol: {args.protocol.upper()} | Split: {args.split.upper()}")
    print("="*65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- AUTO-CHECKPOINT FINDER ---
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint_file = args.checkpoint
    else:
        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            print(f"[!] Cannot find '{checkpoint_dir}/' folder.")
            return
            
        checkpoints = sorted(
            glob.glob(os.path.join(checkpoint_dir, "pgmt_stage1_epoch_*.pth")), 
            key=lambda x: int(os.path.basename(x).split('epoch_')[1].split('.')[0])
        )
        if not checkpoints:
            print(f"[!] No .pth files found in {checkpoint_dir}/")
            return
        checkpoint_file = checkpoints[-1] 
        
    epoch_num = os.path.basename(checkpoint_file).split('epoch_')[1].split('.')[0]
    print(f"[*] Target Checkpoint: {checkpoint_file}")

    # 1. Initialize Dataset
    print(f"[*] Loading Dataset ({args.protocol.upper()} - {args.split.upper()})...")
    dataset = NTURGBD_EvalDataset(
        data_root=args.data_root,
        protocol=args.protocol, 
        split=args.split
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Load Models
    model = PGMTDualStream(vis_dim=1024, stage1_pretraining=True).to(device)
    model = load_model_weights(model, checkpoint_file)
    model.eval()
    
    all_graph_embeddings = []
    all_labels = []
    
    print(f"[*] Extracting Kinematic Graph Features for Target Classes...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            labels = batch["action_label"] 
            
            # Mask to filter only the 10 classes we want to plot
            mask = torch.tensor([label.item() in TARGET_CLASSES for label in labels])
            
            if not mask.any():
                continue 
                
            batch_gpu = {
                "visual_frames_siglip": batch["visual_frames_siglip"][mask].to(device),
                "visual_frames_dinov2": batch["visual_frames_dinov2"][mask].to(device),
                "kinematic_streams": batch["kinematic_streams"][mask].to(device),
                "spatial_anchors": batch["spatial_anchors"][mask].to(device)
            }
            filtered_labels = labels[mask].numpy()
            B_filtered, T = batch_gpu["visual_frames_siglip"].shape[:2]
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(batch_gpu)
                
            # EXPLICITLY EVALUATE THE SEMANTIC SPACE for clean Action Clustering
            graph_feat = outputs["info_nce_graph_sem"] 
            
            # Pool across the Time dimension to get 1 embedding per video
            graph_feat = graph_feat.view(B_filtered, T, -1).mean(dim=1) 
            graph_feat = F.normalize(graph_feat.float(), dim=-1, eps=1e-8)
            
            all_graph_embeddings.append(graph_feat.cpu().numpy())
            all_labels.extend(filtered_labels)

            if (i + 1) % 100 == 0 or (i + 1) == len(dataloader):
                print(f"    - Processed batch {i+1}/{len(dataloader)}")

    # Combine arrays
    if len(all_graph_embeddings) == 0:
        print("[!] No matching classes found in the dataset subset.")
        return

    X = np.vstack(all_graph_embeddings)
    y = np.array(all_labels)
    
    print(f"\n[*] Extracted {X.shape[0]} samples across {len(TARGET_CLASSES)} classes.")
    print("[*] Running t-SNE dimensionality reduction (1024D -> 2D)... This may take a minute.")
    
    # 3. Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X)
    
    # 4. Generate Plot
    print("[*] Plotting Graph Clusters...")
    plt.figure(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(TARGET_CLASSES)))
    
    for idx, (class_id, class_name) in enumerate(TARGET_CLASSES.items()):
        class_mask = (y == class_id)
        plt.scatter(
            X_2d[class_mask, 0], 
            X_2d[class_mask, 1], 
            c=[colors[idx]], 
            label=class_name, 
            alpha=0.7, 
            edgecolors='w', 
            s=80
        )
        
    title_str = f"PGMT Semantic Grounding: MS-HyperTR (Epoch {epoch_num} | {args.protocol.upper()} {args.split.upper()})"
    plt.title(title_str, fontsize=16, pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # File naming based on protocol and split
    file_suffix = f"{args.protocol}_{args.split}_epoch_{epoch_num}"
    png_path = f"outputs/tsne_{file_suffix}.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Export Raw Data to CSV
    csv_path = f"outputs/tsne_{file_suffix}.csv"
    print(f"[*] Exporting raw coordinate data to CSV...")
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X_Coordinate", "Y_Coordinate", "Label_ID", "Class_Name"])
        for idx in range(X_2d.shape[0]):
            x_coord = X_2d[idx, 0]
            y_coord = X_2d[idx, 1]
            label_id = y[idx]
            class_name = TARGET_CLASSES[label_id]
            writer.writerow([x_coord, y_coord, label_id, class_name])
            
    print(f"[*] SUCCESS! Quick-Review PNG saved to: {png_path}")
    print(f"[*] SUCCESS! Raw Data CSV saved to: {csv_path}")
    print("="*65)

if __name__ == "__main__":
    main()