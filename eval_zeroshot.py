import os
import argparse
# CACHE FIX: Force Hugging Face and PyTorch Hub to use local folders
os.environ["HF_HOME"] = "/pnr/lab/nurul/PGMT_Code/hf_cache"
os.environ["TORCH_HOME"] = "/pnr/lab/nurul/PGMT_Code/torch_cache"

import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, SiglipTextModel

from dataset_ntu_eval import NTURGBD_EvalDataset
from pgmt_architecture import PGMTDualStream

# Official NTU RGB+D 120 Action Classes
NTU120_CLASSES = [
    # Original 60 Classes
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
    # Extended 60 Classes (NTU-120)
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

def get_text_embeddings(device):
    """Generates the grounded text embeddings for all 120 NTU classes."""
    print("[*] Loading SigLIP Text Encoder for Zero-Shot Prompts...")
    tokenizer = AutoTokenizer.from_pretrained("google/siglip-large-patch16-384")
    text_model = SiglipTextModel.from_pretrained("google/siglip-large-patch16-384").to(device)
    text_model.eval()

    # Formulate prompts
    prompts = [f"A video of a person {action}." for action in NTU120_CLASSES]
    
    with torch.no_grad():
        inputs = tokenizer(prompts, padding="max_length", return_tensors="pt", truncation=True).to(device)
        text_features = text_model(**inputs).pooler_output
        text_features = F.normalize(text_features, dim=-1, eps=1e-8)
        
    return text_features

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
    parser.add_argument("--no_pool_temporal", action="store_true",
                        help="Disable mean-pooling across temporal frames.")
    args = parser.parse_args()

    pool_temporal = not args.no_pool_temporal

    print("="*65)
    print(" PGMT Phase 1: 120-Class Zero-Shot Text Retrieval")
    print(f" Protocol: {args.protocol.upper()} | Split: {args.split.upper()} | Temporal Pooling: {pool_temporal}")
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
    print(f"[*] Loading Dual-Loss Architecture...")
    model = PGMTDualStream(vis_dim=1024, stage1_pretraining=True).to(device)
    model = load_model_weights(model, checkpoint_file)
    model.eval()
    
    text_embeddings = get_text_embeddings(device) # Shape: (120, 1024)

    total_samples = 0
    top1_correct = 0
    top5_correct = 0
    
    print(f"[*] Extracting Kinematic Graph Features & Computing Text Similarity...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_gpu = {
                "visual_frames_siglip": batch["visual_frames_siglip"].to(device),
                "visual_frames_dinov2": batch["visual_frames_dinov2"].to(device),
                "kinematic_streams": batch["kinematic_streams"].to(device),
                "spatial_anchors": batch["spatial_anchors"].to(device)
            }
            labels = batch["action_label"].to(device) # Shape: (B,)
            B, T = batch["visual_frames_siglip"].shape[:2]
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(batch_gpu)
                
            # EXPLICITLY EVALUATE THE SEMANTIC SPACE
            # This space is perfectly anchored to the pure SigLIP text representation
            graph_feat = outputs["info_nce_graph_sem"]
            
            if pool_temporal:
                # Pool across the Time dimension to get 1 embedding per video
                graph_feat = graph_feat.view(B, T, -1).mean(dim=1) # Shape: (B, 1024)
            else:
                # Per-frame mapping (increases sequence evaluation count)
                graph_feat = graph_feat.view(B * T, -1)
                labels = labels.repeat_interleave(T)
                
            graph_feat = F.normalize(graph_feat.float(), dim=-1, eps=1e-8)
            
            # Compute Cosine Similarity against all 120 Text Prompts
            # text_embeddings shape: (120, 1024) -> similarity shape: (B, 120)
            logits = torch.matmul(graph_feat, text_embeddings.T) 
            
            # Top-K Calculation
            _, top5_indices = logits.topk(5, dim=-1, largest=True, sorted=True)
            
            for j in range(logits.shape[0]):
                true_label = labels[j].item()
                if true_label >= 120: 
                    continue # Safety check
                
                total_samples += 1
                if top5_indices[j][0].item() == true_label:
                    top1_correct += 1
                if true_label in top5_indices[j].tolist():
                    top5_correct += 1

            if (i + 1) % 100 == 0 or (i + 1) == len(dataloader):
                print(f"    - Processed batch {i + 1}/{len(dataloader)}")

    # Final Metrics
    if total_samples > 0:
        top1_acc = (top1_correct / total_samples) * 100.0
        top5_acc = (top5_correct / total_samples) * 100.0
    else:
        top1_acc = top5_acc = 0.0

    print("\n" + "="*65)
    print(" ZERO-SHOT TEXT-TO-SKELETON RETRIEVAL RESULTS (NTU-120)")
    print("="*65)
    print(f"Dataset Pool: Protocol {args.protocol.upper()} | Split {args.split.upper()}")
    print(f"Latent Space: Pure Semantic Dual-Loss Anchor")
    print(f"Total Evaluated Sequences: {total_samples}")
    print(f"\n  Top-1 Zero-Shot Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Zero-Shot Accuracy: {top5_acc:.2f}%")
    print("="*65)

if __name__ == "__main__":
    main()