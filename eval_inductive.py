import os
import sys
import gc
import glob
import argparse
import logging

# CACHE FIX: Force Hugging Face and PyTorch Hub to use local folders on NAS
os.environ["HF_HOME"] = "/pnr/lab/nurul/PGMT_Code/hf_cache"
os.environ["TORCH_HOME"] = "/pnr/lab/nurul/PGMT_Code/torch_cache"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset_split import NTURGBD_SplitDataset
from pgmt_split_architecture import PGMTSplitStream

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_model_weights(model, checkpoint_path):
    # FIX: Safely extract ONLY the 'model' dictionary from the new 125MB resume-ready checkpoints
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Check if it's a nested dictionary (new format) or flat weights (old format)
    state_dict = checkpoint.get('model', checkpoint)
    
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('_orig_mod.', '', 1).replace('module.', '', 1)
        clean_state_dict[clean_k] = v
    model.load_state_dict(clean_state_dict, strict=False)
    return model

@torch.no_grad()
def extract_chunk_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Extracting Features", leave=False, ascii=True):
        B, T = batch["geo_fused"].shape[:2]
        dummy_anchors = torch.zeros(B, T, 2, 2, device=device)
        
        input_dict = {
            "geo_fused": batch["geo_fused"].to(device, non_blocking=True),
            "sem_global": batch["sem_global"].to(device, non_blocking=True),
            "kinematic_streams": batch["kinematic_streams"].to(device, non_blocking=True),
            "spatial_anchors": dummy_anchors
        }
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_dict)
            
            # DIMENSION FIX: Reshape from (B*T, D) back to (B, T, D) and mean-pool
            geo_feat = outputs["info_nce_graph_geo"].view(B, T, -1).mean(dim=1)
            sem_feat = outputs["info_nce_graph_sem"].view(B, T, -1).mean(dim=1)
            
        fused_raw = torch.cat([geo_feat, sem_feat], dim=-1).float()
        fused_raw = torch.nan_to_num(fused_raw, nan=0.0, posinf=0.0, neginf=0.0)
        fused_feat = F.normalize(fused_raw, p=2, dim=-1, eps=1e-8)
        fused_feat = torch.nan_to_num(fused_feat, nan=0.0, posinf=0.0, neginf=0.0)
            
        all_features.append(fused_feat.cpu())
        all_labels.append(batch["action_label"])
        
    if not all_features:
        return torch.empty(0), torch.empty(0)
        
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

def evaluate_knn(train_features, train_labels, test_features, test_labels, device, k=1, batch_size=1024):
    logger.info(f"[*] Starting Batched {k}-NN Evaluation...")
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)
    
    all_preds = []
    
    for i in range(0, test_features.shape[0], batch_size):
        test_batch = test_features[i:i+batch_size].to(device)
        
        similarity = torch.matmul(test_batch, train_features.T)
        topk_sim, topk_indices = similarity.topk(k, dim=1)
        topk_labels = train_labels[topk_indices]
        
        if k == 1:
            batch_preds = topk_labels.squeeze(1)
        else:
            batch_preds, _ = torch.mode(topk_labels, dim=1)
            
        all_preds.append(batch_preds)
        
    preds = torch.cat(all_preds, dim=0)
    acc = (preds == test_labels).float().mean().item() * 100
    logger.info(f"[+] {k}-NN Accuracy: {acc:.2f}%")
    return acc

def evaluate_linear_probe(train_features, train_labels, test_features, test_labels, device, epochs=100):
    logger.info(f"[*] Starting Linear Probing ({epochs} Epochs)...")
    
    embed_dim = train_features.shape[1]
    num_classes = 120
    
    classifier = nn.Linear(embed_dim, num_classes).to(device)
    nn.init.zeros_(classifier.weight)
    nn.init.zeros_(classifier.bias)
    
    optimizer = optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, pin_memory=True, num_workers=0)
    
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    classifier.train()
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = classifier(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(test_features)
        preds = test_logits.argmax(dim=1)
        acc = (preds == test_labels).float().mean().item() * 100
        
    logger.info(f"[+] Linear Probe Accuracy: {acc:.2f}%")
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=str, required=True, choices=["xsub", "xset"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--skeleton_dir", type=str, default="/pnr/lab/nurul/datasets/ntu_rgbd/skeleton")
    parser.add_argument("--offline_features_dir", type=str, default="/pnr/lab/nurul/PGMT_Code/offline_features")
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"[*] Starting {args.protocol.upper()} evaluation on {device} via NAS...")
    
    model = PGMTSplitStream(vis_dim=1024, stage1_pretraining=True)
    model = load_model_weights(model, args.checkpoint)
    model = model.to(device)
    
    chunk_files = sorted(glob.glob(os.path.join(args.offline_features_dir, "ntu_features_chunk_*.pt")))
    if not chunk_files:
        logger.error(f"[!] No feature chunks found in {args.offline_features_dir}. Did you extract them?")
        sys.exit(1)
        
    train_feats_list, train_labels_list = [], []
    test_feats_list, test_labels_list = [], []
    
    for chunk_idx, chunk_file in enumerate(chunk_files):
        logger.info(f"Loading NAS Chunk {chunk_idx + 1}/{len(chunk_files)} into RAM...")
        chunk_data = torch.load(chunk_file, map_location="cpu", weights_only=True)
        
        train_dataset = NTURGBD_SplitDataset(
            chunk_data=chunk_data, skeleton_dir=args.skeleton_dir, 
            protocol=args.protocol, split="train", zsl_classes=0
        )
        if len(train_dataset) > 0:
            train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
            feats, labels = extract_chunk_features(model, train_loader, device)
            train_feats_list.append(feats)
            train_labels_list.append(labels)
            
        test_dataset = NTURGBD_SplitDataset(
            chunk_data=chunk_data, skeleton_dir=args.skeleton_dir, 
            protocol=args.protocol, split="test", zsl_classes=0
        )
        if len(test_dataset) > 0:
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
            feats, labels = extract_chunk_features(model, test_loader, device)
            test_feats_list.append(feats)
            test_labels_list.append(labels)
            
        del chunk_data, train_dataset, test_dataset
        gc.collect()
        
    del model
    torch.cuda.empty_cache()
    
    X_train = torch.cat(train_feats_list, dim=0)
    y_train = torch.cat(train_labels_list, dim=0)
    X_test = torch.cat(test_feats_list, dim=0)
    y_test = torch.cat(test_labels_list, dim=0)
    
    logger.info(f"Data Aggregation Complete. Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    X_train = torch.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = torch.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    evaluate_knn(X_train, y_train, X_test, y_test, device, k=1, batch_size=1024)
    evaluate_linear_probe(X_train, y_train, X_test, y_test, device, epochs=100)

if __name__ == "__main__":
    main()