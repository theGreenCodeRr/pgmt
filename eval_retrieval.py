import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_ntu import NTURGBD_AlignedDataset
from pgmt_architecture import PGMTDualStream
import os

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
    checkpoint_file = "pgmt_stage1_epoch_1.pth" # Update this to your desired epoch

    if not os.path.exists(checkpoint_file):
        print(f"[!] Cannot find {checkpoint_file}. Wait for the training epoch to finish!")
        return

    # 1. Initialize Dataset
    print("[*] Loading NTU Dataset...")
    dataset = NTURGBD_AlignedDataset(data_root="/pnr/lab/nurul/datasets/ntu_rgbd")
    
    # We can use a large batch size here since we aren't accumulating gradients
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    # 2. Initialize Model
    print(f"[*] Loading Weights from {checkpoint_file}...")
    model = PGMTDualStream(vis_dim=1024, stage1_pretraining=True).to(device)
    model = load_model_weights(model, checkpoint_file)
    model.eval()

    all_visual_features = []
    all_graph_features = []
    
    num_eval_batches = 10 # Testing 1,280 continuous video sequences (128 * 10)
    
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
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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