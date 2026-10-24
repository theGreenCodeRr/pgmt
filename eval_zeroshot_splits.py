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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, SiglipTextModel
from tqdm import tqdm

from dataset_split import NTURGBD_SplitDataset
from pgmt_split_architecture import PGMTSplitStream

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Official NTU RGB+D 120 Action Classes
NTU120_CLASSES = [
    "drink water", "eat meal or snack", "brushing teeth", "brushing hair", "drop",
    "pickup", "throw", "sitting down", "standing up", "clapping",
    "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat or cap",
    "take off a hat or cap", "cheer up", "hand waving", "kicking something", "reach into pocket",
    "hopping", "jump up", "make a phone call", "playing with phone or tablet", "typing on a keyboard",
    "pointing to something with finger", "taking a selfie", "check time", "rub two hands together", "nod head or bow",
    "shake head", "wipe face", "salute", "put the palms together", "cross hands in front",
    "sneeze or cough", "staggering", "falling", "touch head", "touch chest",
    "touch back", "touch neck", "nausea or vomiting condition", "use a fan", "punching or slapping other person",
    "kicking other person", "pushing other person", "pat on back of other person", "point finger at the other person", "hugging other person",
    "giving something to other person", "touch other person's pocket", "handshaking", "walking towards each other", "walking apart from each other",
    "put on headphone", "take off headphone", "shoot at the basket", "bounce ball", "tennis bat swing",
    "juggling table tennis balls", "hush", "flick hair", "thumb up", "thumb down",
    "make ok sign", "make victory sign", "staple", "counting money", "cutting nails",
    "cutting paper", "snapping fingers", "open bottle", "sniff", "squat down",
    "toss a coin", "fold paper", "ball up paper", "play magic cube", "apply cream on face",
    "apply cream on hand back", "put on bag", "take off bag", "put something into a bag", "take something out of a bag",
    "open a box", "move heavy objects", "shake fist", "throw up cap or hat", "hands up",
    "cross arms", "arm circles", "arm swings", "running on the spot", "butt kicks",
    "cross toe touch", "side kick", "yawn", "stretch oneself", "blow nose",
    "hit other person with something", "wield knife towards other person", "knock over other person", "grab other person's stuff", "shoot at other person with a gun",
    "step on foot", "high-five", "cheers and drink", "carry something with other person", "take a photo of other person",
    "follow other person", "whisper in other person's ear", "exchange things with other person", "support somebody with hand", "rock-paper-scissors"
]

def get_unseen_indices(split_type):
    if split_type == "110_10":
        return list(range(110, 120))
    elif split_type == "96_24":
        return list(range(96, 120))
    else:
        raise ValueError("Invalid split_type. Choose '110_10' or '96_24'.")

def load_model_weights(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('_orig_mod.', '', 1).replace('module.', '', 1)
        clean_state_dict[clean_k] = v
    model.load_state_dict(clean_state_dict, strict=False)
    return model

def get_unseen_text_embeddings(device, unseen_indices):
    logger.info("[*] Generating SigLIP Text Embeddings for Unseen Zero-Shot Prompts...")
    tokenizer = AutoTokenizer.from_pretrained("google/siglip-large-patch16-384")
    text_model = SiglipTextModel.from_pretrained("google/siglip-large-patch16-384").to(device)
    text_model.eval()

    unseen_prompts = [f"A video of a person {NTU120_CLASSES[i]}." for i in unseen_indices]
    
    with torch.no_grad():
        inputs = tokenizer(unseen_prompts, padding="max_length", return_tensors="pt", truncation=True).to(device)
        text_features = text_model(**inputs).pooler_output
        text_features = F.normalize(text_features, dim=-1, eps=1e-8)
        
    return text_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_type", type=str, required=True, choices=["110_10", "96_24"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--skeleton_dir", type=str, default="/pnr/lab/nurul/datasets/ntu_rgbd/skeleton")
    parser.add_argument("--offline_features_dir", type=str, default="/pnr/lab/nurul/PGMT_Code/offline_features")
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info(f" PGMT Phase 1: Inductive Zero-Shot Split Evaluation ({args.split_type.replace('_', '/')})")
    logger.info("=" * 65)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unseen_indices = get_unseen_indices(args.split_type)
    zsl_count = 10 if args.split_type == "110_10" else 24

    logger.info(f"[*] Loading Dual-Stream Architecture from {args.checkpoint}...")
    model = PGMTSplitStream(vis_dim=1024, stage1_pretraining=True).to(device)
    model = load_model_weights(model, args.checkpoint)
    model.eval()
    
    unseen_text_embeddings = get_unseen_text_embeddings(device, unseen_indices) 
    unseen_text_embeddings = torch.nan_to_num(unseen_text_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    chunk_files = sorted(glob.glob(os.path.join(args.offline_features_dir, "ntu_features_chunk_*.pt")))
    if not chunk_files:
        logger.error(f"[!] No feature chunks found in {args.offline_features_dir}.")
        sys.exit(1)

    total_samples = 0
    top1_correct = 0
    top5_correct = 0

    logger.info(f"[*] Extracting Kinematics & Computing Strict Inductive ZSL Alignment from NAS Chunks...")

    with torch.no_grad():
        for chunk_idx, chunk_file in enumerate(chunk_files):
            logger.info(f"Processing NAS Chunk {chunk_idx + 1}/{len(chunk_files)}")
            chunk_data = torch.load(chunk_file, map_location="cpu", weights_only=True)
            
            dataset = NTURGBD_SplitDataset(
                chunk_data=chunk_data, 
                skeleton_dir=args.skeleton_dir, 
                protocol="xsub", 
                split="test", 
                zsl_classes=zsl_count
            )
            
            if len(dataset) == 0:
                del chunk_data, dataset
                gc.collect()
                continue

            dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

            for batch in tqdm(dataloader, desc=f"Chunk {chunk_idx + 1}", leave=False, ascii=True):
                valid_mask = torch.tensor([label.item() in unseen_indices for label in batch["action_label"]])
                if not valid_mask.any():
                    continue
                    
                B_valid = valid_mask.sum().item()
                T = batch["geo_fused"].shape[1]
                dummy_anchors = torch.zeros(B_valid, T, 2, 2, device=device)
                
                input_dict = {
                    "geo_fused": batch["geo_fused"][valid_mask].to(device, non_blocking=True),
                    "sem_global": batch["sem_global"][valid_mask].to(device, non_blocking=True),
                    "kinematic_streams": batch["kinematic_streams"][valid_mask].to(device, non_blocking=True),
                    "spatial_anchors": dummy_anchors
                }
                labels = batch["action_label"][valid_mask]
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(input_dict)
                    
                    # DIMENSION FIX: Reshape from (B*T, D) back to (B, T, D) and mean-pool
                    graph_feat = outputs["info_nce_graph_sem"].view(B_valid, T, -1).mean(dim=1)
                
                graph_feat = graph_feat.float()
                graph_feat = torch.nan_to_num(graph_feat, nan=0.0, posinf=0.0, neginf=0.0)
                graph_feat = F.normalize(graph_feat, p=2, dim=-1, eps=1e-8)
                graph_feat = torch.nan_to_num(graph_feat, nan=0.0, posinf=0.0, neginf=0.0)
                
                logits = torch.matmul(graph_feat, unseen_text_embeddings.T) 
                k_val = min(5, logits.shape[1])
                _, topk_indices = logits.topk(k_val, dim=-1, largest=True, sorted=True)
                
                for j in range(logits.shape[0]):
                    true_label_global = labels[j].item()
                    true_label_local = unseen_indices.index(true_label_global)
                    
                    total_samples += 1
                    if topk_indices[j][0].item() == true_label_local:
                        top1_correct += 1
                    if true_label_local in topk_indices[j].tolist():
                        top5_correct += 1
            
            del chunk_data, dataset
            gc.collect()

    top1_acc = (top1_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    top5_acc = (top5_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

    logger.info("\n" + "=" * 65)
    logger.info(f" INDUCTIVE ZERO-SHOT RESULTS ({args.split_type.replace('_', '/')})")
    logger.info("=" * 65)
    logger.info(f"Total Evaluated Unseen Sequences: {total_samples}")
    logger.info(f"  Top-1 Zero-Shot Accuracy: {top1_acc:.2f}%")
    logger.info(f"  Top-5 Zero-Shot Accuracy: {top5_acc:.2f}%")
    logger.info("=" * 65)

if __name__ == "__main__":
    main()
