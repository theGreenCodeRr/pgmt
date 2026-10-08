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
    """Safely loads weights, stripping DDP/Compile prefixes."""
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith('_orig_mod.'):
            clean_k = clean_k.replace('_orig_mod.', '', 1)
        if clean_k.startswith('module.'):
            clean_k = clean_k.replace('module.', '', 1)
        clean_state_dict[clean_k] = v
    model.load_state_dict(clean_state_dict, strict=False)
    return model

def compute_chunked_retrieval(query_pool, target_pool, topk=(1, 5, 10), chunk_size=2048, device="cuda"):
    """
    Memory-Safe Top-K Retrieval.
    Calculates the N x N similarity matrix in chunks to prevent VRAM OOM errors
    when evaluating massive datasets across 114k instances.
    """
    query_pool = query_pool.to(device)
    target_pool = target_pool.to(device)
    N = query_pool.shape[0]
    max_k = max(topk)
    
    correct_counts = {k: 0 for k in topk}
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    
    # Process the similarity matrix in vertical slices (chunks)
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        query_chunk = query_pool[i:end]  # Shape: (Chunk, 1024)
        
        # Calculate cosine similarity for this slice: (Chunk, N)
        sim_chunk = torch.matmul(query_chunk, target_pool.T)
        
        # 1. Calculate Top-K Matches
        _, topk_indices = sim_chunk.topk(max_k, dim=-1, largest=True, sorted=True)
        true_indices = torch.arange(i, end, device=device).view(-1, 1)
        
        for k in topk:
            correct = topk_indices[:, :k].eq(true_indices).sum().item()
            correct_counts[k] += correct
            
        # 2. Positives vs Negatives Similarity Separation
        pos_sim_chunk = sim_chunk[torch.arange(end - i, device=device), torch.arange(i, end, device=device)]
        total_pos_sim += pos_sim_chunk.sum().item()
        total_neg_sim += (sim_chunk.sum().item() - pos_sim_chunk.sum().item())
        
    accuracies = {f"Top-{k}": (correct_counts[k] / N) * 100.0 for k in topk}
    pos_sim_avg = total_pos_sim / N
    neg_sim_avg = total_neg_sim / (N * (N - 1)) if N > 1 else 0.0
    
    return accuracies, pos_sim_avg, neg_sim_avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/pnr/lab/nurul/datasets/ntu_rgbd")
    parser.add_argument("--protocol", type=str, default="all", choices=["all", "xsub", "xset", "xview"],
                        help="Partition protocol. Use 'all' for full uncurated dataset.")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "test"],
                        help="Data split. Use 'all' for full uncurated dataset.")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Explicit path to checkpoint file (e.g., checkpoints/pgmt_stage1_epoch_160.pth)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--no_pool_temporal", action="store_true",
                        help="Disable mean-pooling across temporal frames (evaluates per-frame instance matching).")
    args = parser.parse_args()

    pool_temporal = not args.no_pool_temporal

    print("=" * 65)
    print(f" PGMT Phase 1: Instance Retrieval Evaluation")
    print(f" Protocol: {args.protocol.upper()} | Split: {args.split.upper()} | Temporal Pooling: {pool_temporal}")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Resolve Checkpoint
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
            print(f"[!] No checkpoints found in {checkpoint_dir}/")
            return
        checkpoint_file = checkpoints[-1]
        
    print(f"[*] Target Checkpoint: {checkpoint_file}")

    # 2. Initialize Evaluation Dataset
    print(f"[*] Loading Dataset ({args.protocol.upper()} - {args.split.upper()})...")
    dataset = NTURGBD_EvalDataset(
        data_root=args.data_root,
        protocol=args.protocol, 
        split=args.split
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 3. Load Model
    print(f"[*] Loading Dual-Loss Architecture...")
    model = PGMTDualStream(vis_dim=1024, stage1_pretraining=True).to(device)
    model = load_model_weights(model, checkpoint_file)
    model.eval()

    all_visual_features = []
    all_graph_features = []
    
    print(f"[*] Extracting Geometric Spatial Features across {len(dataset)} sequences...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_gpu = {
                "visual_frames_siglip": batch["visual_frames_siglip"].to(device),
                "visual_frames_dinov2": batch["visual_frames_dinov2"].to(device),
                "kinematic_streams": batch["kinematic_streams"].to(device),
                "spatial_anchors": batch["spatial_anchors"].to(device)
            }
            B, T = batch["visual_frames_siglip"].shape[:2]
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(batch_gpu)
                
            # Geometric Grounding Output: (B*T, 1024)
            vis_feat = outputs["info_nce_visual_geo"]
            graph_feat = outputs["info_nce_graph_geo"]
            
            if pool_temporal:
                # Average local actor patch tokens across T frames: (B, T, 1024) -> (B, 1024)
                vis_feat = vis_feat.view(B, T, -1).mean(dim=1)
                graph_feat = graph_feat.view(B, T, -1).mean(dim=1)
            
            vis_feat = F.normalize(vis_feat.float(), dim=-1, eps=1e-8)
            graph_feat = F.normalize(graph_feat.float(), dim=-1, eps=1e-8)
            
            # Transfer to host CPU memory to conserve VRAM during large 114k gathering
            all_visual_features.append(vis_feat.cpu())
            all_graph_features.append(graph_feat.cpu())
            
            if (i + 1) % 100 == 0 or (i + 1) == len(dataloader):
                print(f"    - Processed batch {i + 1}/{len(dataloader)}")

    # 4. Concatenate Pools
    visual_pool = torch.cat(all_visual_features, dim=0)
    graph_pool = torch.cat(all_graph_features, dim=0)
    N = visual_pool.shape[0]

    print(f"\n[*] Generated Retrieval Pool of {N} multimodal instances.")
    print(f"[*] Computing Chunked Cross-Modal Similarity Matrix on {device}...")
    
    g2v_acc, pos_sim, neg_sim = compute_chunked_retrieval(
        query_pool=graph_pool, target_pool=visual_pool, chunk_size=args.chunk_size, device=device
    )
    v2g_acc, _, _ = compute_chunked_retrieval(
        query_pool=visual_pool, target_pool=graph_pool, chunk_size=args.chunk_size, device=device
    )

    print("\n" + "=" * 65)
    print(f" INSTANCE RETRIEVAL RESULTS: {args.protocol.upper()} ({args.split.upper()})")
    print("=" * 65)
    print(f"Candidate Gallery Size: {N} instances")
    print(f"Temporal Aggregation: {'Mean-pooled across frames' if pool_temporal else 'Per-frame anchor matching'}\n")
    
    print("Graph-to-Vision Retrieval (Given Skeleton -> Find Pixels):")
    print(f"  Top-1 Accuracy:  {g2v_acc['Top-1']:.2f}%")
    print(f"  Top-5 Accuracy:  {g2v_acc['Top-5']:.2f}%")
    print(f"  Top-10 Accuracy: {g2v_acc['Top-10']:.2f}%\n")
    
    print("Vision-to-Graph Retrieval (Given Pixels -> Find Skeleton):")
    print(f"  Top-1 Accuracy:  {v2g_acc['Top-1']:.2f}%")
    print(f"  Top-5 Accuracy:  {v2g_acc['Top-5']:.2f}%")
    print(f"  Top-10 Accuracy: {v2g_acc['Top-10']:.2f}%\n")
    
    print("Cosine Similarity Distribution:")
    print(f"  Avg Similarity (True Matches): {pos_sim:.4f}")
    print(f"  Avg Similarity (Distractors):  {neg_sim:.4f}")
    print(f"  Separation Margin:             {pos_sim - neg_sim:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    main()