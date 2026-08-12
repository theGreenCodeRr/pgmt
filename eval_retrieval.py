import os
import argparse
# CACHE FIX: Force Hugging Face and PyTorch Hub to use local folders
os.environ["HF_HOME"] = "/pnr/lab/nurul/PGMT_Code/hf_cache"
os.environ["TORCH_HOME"] = "/pnr/lab/nurul/PGMT_Code/torch_cache"

import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_ntu_eval import NTURGBD_EvalDataset
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
    # strict=False allows us to load the tiny 20MB file while keeping Google/Meta backbones frozen
    model.load_state_dict(clean_state_dict, strict=False)
    return model

def compute_chunked_retrieval(query_pool, target_pool, topk=(1, 5, 10), chunk_size=2048):
    """
    Memory-Safe Top-K Retrieval.
    Calculates the N x N similarity matrix in chunks to prevent VRAM OOM errors
    when evaluating massive datasets.
    """
    N = query_pool.shape[0]
    device = query_pool.device
    max_k = max(topk)
    
    correct_counts = {k: 0 for k in topk}
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    
    # Process the similarity matrix in vertical slices (chunks)
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        query_chunk = query_pool[i:end] # Shape: (Chunk, 1024)
        
        # Calculate similarity for just this chunk -> Shape: (Chunk, N)
        sim_chunk = torch.matmul(query_chunk, target_pool.T)
        
        # 1. Calculate Top-K
        _, topk_indices = sim_chunk.topk(max_k, dim=-1, largest=True, sorted=True)
        true_indices = torch.arange(i, end, device=device).view(-1, 1)
        
        for k in topk:
            correct = topk_indices[:, :k].eq(true_indices).sum().item()
            correct_counts[k] += correct
            
        # 2. Calculate Similarity Gap (Positives vs Negatives)
        pos_sim_chunk = sim_chunk[torch.arange(end - i), torch.arange(i, end)]
        total_pos_sim += pos_sim_chunk.sum().item()
        
        total_neg_sim += (sim_chunk.sum().item() - pos_sim_chunk.sum().item())
        
    accuracies = {f"Top-{k}": (correct_counts[k] / N) * 100.0 for k in topk}
    pos_sim_avg = total_pos_sim / N
    neg_sim_avg = total_neg_sim / (N * (N - 1)) if N > 1 else 0.0
    
    return accuracies, pos_sim_avg, neg_sim_avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=str, default="xsub", choices=["xsub", "xview"], 
                        help="Choose 'xsub' (Cross-Subject) or 'xview' (Cross-View)")
    args = parser.parse_args()

    print("="*65)
    print(f" PGMT Phase 1: {args.protocol.upper()} Instance Retrieval Evaluation")
    print("="*65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- AUTO-CHECKPOINT FINDER ---
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print(f"[!] Cannot find '{checkpoint_dir}/' folder. Wait for the training epoch to finish!")
        return
        
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "pgmt_stage1_epoch_*.pth")), 
        key=lambda x: int(os.path.basename(x).split('epoch_')[1].split('.')[0])
    )
    
    if not checkpoints:
        print(f"[!] No .pth files found in {checkpoint_dir}/")
        return
        
    checkpoint_file = checkpoints[-1] 
    print(f"[*] Found {len(checkpoints)} checkpoints. Using the latest: {checkpoint_file}")

    # 1. Initialize Dataset using the Argparse Protocol
    print(f"[*] Loading NTU {args.protocol.upper()} Test Dataset...")
    dataset = NTURGBD_EvalDataset(
        data_root="/pnr/lab/nurul/datasets/ntu_rgbd",
        protocol=args.protocol, 
        split="test"
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 2. Load Models
    print(f"[*] Loading Dual-Loss Architecture Weights...")
    model = PGMTDualStream(vis_dim=1024, stage1_pretraining=True).to(device)
    model = load_model_weights(model, checkpoint_file)
    model.eval()

    all_visual_features = []
    all_graph_features = []
    
    print(f"[*] Extracting Geometric Spatial Features for Instance Retrieval...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_gpu = {
                "visual_frames_siglip": batch["visual_frames_siglip"].to(device),
                "visual_frames_dinov2": batch["visual_frames_dinov2"].to(device),
                "kinematic_streams": batch["kinematic_streams"].to(device),
                "spatial_anchors": batch["spatial_anchors"].to(device)
            }
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(batch_gpu)
                
            # EXPLICITLY EVALUATE THE GEOMETRIC SPACE
            # Instance matching requires the 3D depth and room structure of DINOv2
            vis_feat = outputs["info_nce_visual_geo"]
            graph_feat = outputs["info_nce_graph_geo"]
            
            vis_feat = F.normalize(vis_feat.float(), dim=-1, eps=1e-8)
            graph_feat = F.normalize(graph_feat.float(), dim=-1, eps=1e-8)
            
            all_visual_features.append(vis_feat)
            all_graph_features.append(graph_feat)
            
            if (i + 1) % 50 == 0:
                print(f"    - Processed batch {i+1}/{len(dataloader)}")

    # Concatenate all features
    visual_pool = torch.cat(all_visual_features, dim=0) # Shape: (N, 1024)
    graph_pool = torch.cat(all_graph_features, dim=0)   # Shape: (N, 1024)
    N = visual_pool.shape[0]

    print(f"\n[*] Generated {args.protocol.upper()} Retrieval Pool of {N} multimodal pairs.")
    print("[*] Computing Chunked Cross-Modal Similarity Matrix (Memory Safe)...")
    
    # Run Chunked Memory-Safe Evaluation
    g2v_acc, pos_sim, neg_sim = compute_chunked_retrieval(query_pool=graph_pool, target_pool=visual_pool)
    v2g_acc, _, _ = compute_chunked_retrieval(query_pool=visual_pool, target_pool=graph_pool)

    print("\n" + "="*65)
    print(f" OFFICIAL {args.protocol.upper()} EVALUATION RESULTS: 1%, 5%, 10% TABLE")
    print("="*65)
    print(f"Test Protocol: {args.protocol.upper()} (Exact Video Instance Matching)")
    print(f"Latent Space:  Geometric (Fused SigLIP + DINOv2)")
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
    print("="*65)

if __name__ == "__main__":
    main()