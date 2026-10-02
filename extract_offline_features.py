import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Force Hugging Face and PyTorch Hub caches
os.environ["HF_HOME"] = "/pnr/lab/nurul/PGMT_Code/hf_cache"
os.environ["TORCH_HOME"] = "/pnr/lab/nurul/PGMT_Code/torch_cache"

from dataset_ntu import NTURGBD_AlignedDataset
from pgmt_architecture import DualVisionEncoder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Tier 3 Offline Feature Extraction on {device}...")

    # Output directory for the chunked tensor files on the NAS
    output_dir = "/pnr/lab/nurul/PGMT_Code/offline_features"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize Dataset (We use the original dataloader to read the .avi videos one last time)
    dataset = NTURGBD_AlignedDataset(data_root="/pnr/lab/nurul/datasets/ntu_rgbd")
    
    # We use 4 workers to handle the NAS latency safely since we are not calculating gradients
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True, drop_last=False)

    # 2. Initialize only the Frozen Vision Encoders
    vision_encoder = DualVisionEncoder(output_dim=1024).to(device)
    vision_encoder.eval()

    chunk_size = 10000
    current_chunk_idx = 0
    extracted_features = {}

    print(f"[*] Starting extraction of {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting Features")):
            # Skip corrupted/dummy samples injected by the dataset's fallback mechanism
            valid_mask = [sid != "dummy" for sid in batch["sample_id"]]
            if not any(valid_mask):
                continue
                
            rgb_siglip = batch["visual_frames_siglip"][valid_mask].to(device, non_blocking=True)
            rgb_dinov2 = batch["visual_frames_dinov2"][valid_mask].to(device, non_blocking=True)
            valid_sample_ids = [sid for sid in batch["sample_id"] if sid != "dummy"]

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Extract both Geometric (Fused) and Semantic (Global SigLIP) spaces
                vis_tokens_fused, vis_tokens_siglip_global = vision_encoder(rgb_siglip, rgb_dinov2)
                
            # Move to CPU immediately to prevent VRAM accumulation
            vis_tokens_fused = vis_tokens_fused.cpu()
            vis_tokens_siglip_global = vis_tokens_siglip_global.cpu()

            for i, sample_id in enumerate(valid_sample_ids):
                extracted_features[sample_id] = {
                    "geo_fused": vis_tokens_fused[i].clone(),
                    "sem_global": vis_tokens_siglip_global[i].clone()
                }

            # Save in manageable chunks to respect system RAM limits during the subsequent training
            if len(extracted_features) >= chunk_size or (batch_idx == len(dataloader) - 1):
                save_path = os.path.join(output_dir, f"ntu_features_chunk_{current_chunk_idx}.pt")
                torch.save(extracted_features, save_path)
                print(f"\n[*] Saved {len(extracted_features)} features to {save_path}")
                
                # Reset dictionary for the next chunk to free up RAM
                extracted_features = {}
                current_chunk_idx += 1

    print("[*] Tier 3 Offline Feature Extraction Complete.")

if __name__ == "__main__":
    main()