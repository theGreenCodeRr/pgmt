import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_ntu import NTURGBD_AlignedDataset
from architecture import PGMTDualStream
import os

def load_model_weights(model, checkpoint_path):
    """Safely loads weights, stripping the '_orig_mod.' and 'module.' prefixes."""
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
    """
    Computes Top-K retrieval accuracy.
    sim_matrix: (N, N) cosine similarity matrix.
    """
    N = sim_matrix.shape[0]
    target_labels = torch.arange(N, device=sim_matrix.device)
    
    accuracies = {}
    for k in topk:
        # Get the top-k predictions for each row
        _, topk_indices = sim_matrix.topk(k, dim=-1, largest=True, sorted=True)
        # Check if the target label is in the top-k predictions
        correct = topk_indices.eq(target_labels.view(-1, 1).expand_as(topk_indices))
        # Sum the correct predictions and divide by total
        acc = correct.sum().item() / N
        accuracies[f"Top-{k}"] = acc * 100.0
        
    return accuracies

def main():
    print("="*60)
    print(" PGMT Phase 1: Cross-Modal Retrieval Evaluation")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_file = "pgmt_stage1_epoch10.pth"

    if not os.path.exists(checkpoint_file):
        print(f"[!] Cannot find {checkpoint_file}. Please ensure the path is correct.")
        return

    # 1. Initialize Dataset
    print("[*] Loading NTU Dataset...")
    dataset = NTURGBD_AlignedDataset(data_root=r"E:\datasets\ntu_rgbd", num_visual_frames=3)
    
    # Use DataLoader to fetch a continuous evaluation pool
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # 2. Initialize Model
    print(f"[*] Loading Weights from {checkpoint_file}...")
    model = PGMTDualStream(stage1_pretraining=True).to(device)
    model = load_model_weights(model, checkpoint_file)
    model.eval()

    all_visual_features = []
    all_graph_features = []
    
    num_eval_batches = 10 # 10 batches of 32 = 320 videos * 3 frames = 960 embeddings
    
    print(f"[*] Extracting Features for Retrieval Pool ({num_eval_batches} batches)...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_eval_batches:
                break
                
            batch_gpu = {
                "visual_frames": batch["visual_frames"].to(device),
                "kinematic_windows": batch["kinematic_windows"].to(device),
                "spatial_anchors": batch["spatial_anchors"].to(device)
            }
            
            with torch.amp.autocast('cuda'):
                outputs = model(batch_gpu)
                
            vis_feat = outputs["info_nce_visual"]
            graph_feat = outputs["info_nce_graph"]
            
            # L2 Normalize the features for Cosine Similarity
            vis_feat = F.normalize(vis_feat.float(), dim=-1)
            graph_feat = F.normalize(graph_feat.float(), dim=-1)
            
            all_visual_features.append(vis_feat)
            all_graph_features.append(graph_feat)
            
            print(f"    - Processed batch {i+1}/{num_eval_batches}")

    # Concatenate all features to form the retrieval pool
    visual_pool = torch.cat(all_visual_features, dim=0) # Shape: (N, 768)
    graph_pool = torch.cat(all_graph_features, dim=0)   # Shape: (N, 768)
    N = visual_pool.shape[0]

    print(f"\n[*] Generated Retrieval Pool of {N} multimodal pairs.")
    print("[*] Computing Cross-Modal Cosine Similarity Matrix...")
    
    # Calculate N x N similarity matrix
    # sim_matrix[i, j] = similarity between graph_i and visual_j
    sim_matrix = torch.matmul(graph_pool, visual_pool.T)

    # Calculate Accuracies
    # Graph-to-Vision: Given a skeleton, find the image patch
    g2v_acc = calculate_topk_accuracy(sim_matrix, topk=(1, 5, 10))
    # Vision-to-Graph: Given an image patch, find the skeleton
    v2g_acc = calculate_topk_accuracy(sim_matrix.T, topk=(1, 5, 10))

    # Calculate Similarity Gap (Proof of Contrastive Learning Success)
    # Diagonal = True Matches. Off-Diagonal = Distractors
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