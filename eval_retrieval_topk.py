import os
os.environ["HF_HOME"] = "/pnr/lab/nurul/PGMT_Code/hf_cache"
os.environ["TORCH_HOME"] = "/pnr/lab/nurul/PGMT_Code/torch_cache"
os.makedirs("/pnr/lab/nurul/PGMT_Code/hf_cache", exist_ok=True)
os.makedirs("/pnr/lab/nurul/PGMT_Code/torch_cache", exist_ok=True)
os.makedirs("/pnr/lab/nurul/PGMT_Code/logs", exist_ok=True)
os.makedirs("/pnr/lab/nurul/PGMT_Code/checkpoints", exist_ok=True)

import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_ntu import NTURGBD_AlignedDataset
from pgmt_architecture import PGMTDualStream

def load_model_weights(model, checkpoint_path):
    """Safely loads weights, stripping the '_orig_mod.' and 'module.' prefixes from DDP/Compile."""
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith('_orig_mod.'):
            clean_k = clean_k.replace('_orig_mod.', '', 1)
        if clean_k.startswith('module.'):
            clean_k = clean_k.replace('module.', '', 1)
        clean_state_dict[clean_k] = v
    model.load_state_dict(clean_state_dict)
    return model

def calculate_topk_accuracy(sim_matrix, topk=(1, 5, 10)):
    """Computes Top-K retrieval accuracy from an N x N similarity matrix."""
    N = sim_matrix.shape[0]
    target_labels = torch.arange(N, device=sim_matrix.device)
    
    accuracies = {}
    for k in topk:
        _, topk_indices = sim_matrix.topk(k, dim=-1, largest=True, sorted=True)
        correct = topk_indices.eq(target_labels.view(-1, 1).expand_as(topk_indices))
        acc = correct.sum().item() / N
        accuracies[f"Top-{k}"] = acc * 100.0
        
    return accuracies

def main():
    print("="*60)
    print(" PGMT Phase 1: Cross-Modal Retrieval Evaluation (H100 SOTA)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- AUTO-CHECKPOINT FINDER ---
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print(f"[!] Cannot find '{checkpoint_dir}/' folder. Wait for the training epoch to finish!")
        return
        
    # Dynamically find and sort all epoch checkpoints
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "pgmt_stage1_epoch_*.pth")), 
        key=lambda x: int(os.path.basename(x).split('epoch_')[1].split('.')[0])
    )
    
    if not checkpoints:
        print(f"[!] No .pth files found in {checkpoint_dir}/")
        return
        
    checkpoint_file = checkpoints[-1] # Automatically grab the latest epoch
    print(f"[*] Found {len(checkpoints)} checkpoints. Using the latest: {checkpoint_file}")

    # 1. Initialize Dataset
    print("[*] Loading NTU Dataset...")
    dataset = NTURGBD_AlignedDataset(data_root="/pnr/lab/nurul/datasets/ntu_rgbd")
    
    # Lowered batch size to 32 and workers to 2 for safe single-GPU processing
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # 2. Initialize Model
    print(f"[*] Loading Weights from {checkpoint_file}...")
    model = PGMTDualStream(vis_dim=1024, stage1_pretraining=True).to(device)
    model = load_model_weights(model, checkpoint_file)
    model.eval()

    all_visual_features = []
    all_graph_features = []
    
    # Testing 1,280 continuous video sequences (32 batch_size * 40 batches = 1280 pool)
    num_eval_batches = 40 
    
    print(f"[*] Extracting Features for Retrieval Pool ({num_eval_batches} batches)...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_eval_batches:
                break
                
            batch_gpu = {
                "visual_frames_siglip": batch["visual_frames_siglip"].to(device),
                "visual_frames_dinov2": batch["visual_frames_dinov2"].to(device),
                "kinematic_streams": batch["kinematic_streams"].to(device),
                "spatial_anchors": batch["spatial_anchors"].to(device)
            }
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(batch_gpu)
                
            vis_feat = outputs["info_nce_visual"]
            graph_feat = outputs["info_nce_graph"]
            
            # L2 Normalize the features for Cosine Similarity (Anti-NaN Guarded)
            vis_feat = F.normalize(vis_feat.float(), dim=-1, eps=1e-8)
            graph_feat = F.normalize(graph_feat.float(), dim=-1, eps=1e-8)
            
            all_visual_features.append(vis_feat)
            all_graph_features.append(graph_feat)
            
            print(f"    - Processed batch {i+1}/{num_eval_batches}")

    # Concatenate all features to form the retrieval pool
    visual_pool = torch.cat(all_visual_features, dim=0) # Shape: (N, 1024)
    graph_pool = torch.cat(all_graph_features, dim=0)   # Shape: (N, 1024)
    N = visual_pool.shape[0]

    print(f"\n[*] Generated Retrieval Pool of {N} multimodal pairs.")
    print("[*] Computing Cross-Modal Cosine Similarity Matrix...")
    
    # Calculate N x N similarity matrix
    sim_matrix = torch.matmul(graph_pool, visual_pool.T)

    # Calculate Accuracies
    g2v_acc = calculate_topk_accuracy(sim_matrix, topk=(1, 5, 10))
    v2g_acc = calculate_topk_accuracy(sim_matrix.T, topk=(1, 5, 10))

    # Calculate Similarity Gap (Proof of Contrastive Learning Success)
    mask_pos = torch.eye(N, dtype=torch.bool, device=device)
    pos_sim = sim_matrix[mask_pos].mean().item()
    neg_sim = sim_matrix[~mask_pos].mean().item()

    print("\n" + "="*60)
    print(" EVALUATION RESULTS: CROSS-MODAL ALIGNMENT")
    print("="*60)
    print(f"Total Retrieval Pool Size: {N} instances\n")
    
    print("Graph-to-Vision Retrieval (Given Skeleton -> Find Pixels):")
    print(f"  Top-1 Accuracy:  {g2v_acc['Top-1']:.2f}%")
    print(f"  Top-5 Accuracy:  {g2v_acc['Top-5']:.2f}%")
    print(f"  Top-10 Accuracy: {g2v_acc['Top-10']:.2f}%\n")
    
    print("Vision-to-Graph Retrieval (Given Pixels -> Find Skeleton):")
    print(f"  Top-1 Accuracy:  {v2g_acc['Top-1']:.2f}%")
    print(f"  Top-5 Accuracy:  {v2g_acc['Top-5']:.2f}%")
    print(f"  Top-10 Accuracy: {v2g_acc['Top-10']:.2f}%\n")
    
    print("Cosine Similarity Analysis:")
    print(f"  Avg Similarity (True Matches): {pos_sim:.4f}")
    print(f"  Avg Similarity (Distractors):  {neg_sim:.4f}")
    print(f"  Similarity Gap (Margin):       {pos_sim - neg_sim:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()