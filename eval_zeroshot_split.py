import os
# CACHE FIX: Force Hugging Face and PyTorch Hub to use local folders
os.environ["HF_HOME"] = "/pnr/lab/nurul/PGMT_Code/hf_cache"
os.environ["TORCH_HOME"] = "/pnr/lab/nurul/PGMT_Code/torch_cache"

import glob
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, SiglipTextModel

from dataset_ntu_eval import NTURGBD_EvalDataset
from pgmt_architecture import PGMTDualStream

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

# Standard Unseen Splits for NTU-120 ZSL Protocols
UNSEEN_10 = [
    "brushing teeth", "throw", "tear up paper", "take off glasses", "pointing to something with finger",
    "wipe face", "falling", "pushing other person", "walking towards each other", "bounce ball"
]

UNSEEN_24 = UNSEEN_10 + [
    "thumb up", "cutting paper", "sniff", "play magic cube", "take off bag",
    "shake fist", "cross arms", "butt kicks", "yawn", "hit other person with something",
    "step on foot", "take a photo of other person", "whisper in other person's ear", "rock-paper-scissors"
]

def load_model_weights(model, checkpoint_path):
    """Safely loads weights, stripping DDP/Compile prefixes."""
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('_orig_mod.', '').replace('module.', '')
        clean_state_dict[clean_k] = v
    model.load_state_dict(clean_state_dict, strict=False)
    return model

def get_unseen_text_embeddings(unseen_classes, device):
    """Generates text embeddings ONLY for the unseen classes."""
    print(f"[*] Loading SigLIP Text Encoder for {len(unseen_classes)} Unseen Prompts...")
    tokenizer = AutoTokenizer.from_pretrained("google/siglip-large-patch16-384")
    text_model = SiglipTextModel.from_pretrained("google/siglip-large-patch16-384").to(device)
    text_model.eval()

    prompts = [f"A video of a person {action}." for action in unseen_classes]
    
    with torch.no_grad():
        inputs = tokenizer(prompts, padding="max_length", return_tensors="pt", truncation=True).to(device)
        text_features = text_model(**inputs).pooler_output
        text_features = F.normalize(text_features, dim=-1, eps=1e-8)
        
    return text_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="110_10", choices=["110_10", "96_24"], 
                        help="Choose '110_10' or '96_24' zero-shot split")
    args = parser.parse_args()

    print("="*65)
    print(f" PGMT Phase 1: Zero-Shot Text Retrieval (Split: {args.split.replace('_', '/')})")
    print("="*65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define search space
    unseen_class_names = UNSEEN_10 if args.split == "110_10" else UNSEEN_24
    unseen_global_indices = [NTU120_CLASSES.index(cls) for cls in unseen_class_names]
    
    checkpoint_dir = "checkpoints"
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "pgmt_stage1_epoch_*.pth")), 
        key=lambda x: int(os.path.basename(x).split('epoch_')[1].split('.')[0])
    )
    if not checkpoints:
        print(f"[!] No .pth files found in {checkpoint_dir}/")
        return
    checkpoint_file = checkpoints[-1] 

    print("[*] Loading NTU-120 X-Sub Test Dataset...")
    dataset = NTURGBD_EvalDataset(
        data_root="/pnr/lab/nurul/datasets/ntu_rgbd",
        protocol="xsub", 
        split="test"
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Load Model
    model = PGMTDualStream(vis_dim=1024, stage1_pretraining=True).to(device)
    model = load_model_weights(model, checkpoint_file)
    model.eval()
    
    # Generate the restricted search space
    text_embeddings = get_unseen_text_embeddings(unseen_class_names, device) # Shape: (U, 1024)

    total_samples = 0
    top1_correct = 0
    top5_correct = 0
    
    print(f"[*] Evaluating exclusively on the {len(unseen_class_names)} unseen classes...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            labels = batch["action_label"] # Shape: (B,)
            
            # Mask to filter ONLY sequences belonging to the Unseen classes
            mask = torch.tensor([label.item() in unseen_global_indices for label in labels])
            if not mask.any():
                continue 
                
            batch_gpu = {
                "visual_frames_siglip": batch["visual_frames_siglip"][mask].to(device),
                "visual_frames_dinov2": batch["visual_frames_dinov2"][mask].to(device),
                "kinematic_streams": batch["kinematic_streams"][mask].to(device),
                "spatial_anchors": batch["spatial_anchors"][mask].to(device)
            }
            
            # Map the global 120-class labels to the local U-class indices (0 to U-1)
            filtered_global_labels = labels[mask].numpy()
            local_labels = [unseen_global_indices.index(l) for l in filtered_global_labels]
            
            B_filtered, T = batch_gpu["visual_frames_siglip"].shape[:2]
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(batch_gpu)
                
            graph_feat = outputs["info_nce_graph_sem"]
            graph_feat = graph_feat.view(B_filtered, T, -1).mean(dim=1) 
            graph_feat = F.normalize(graph_feat.float(), dim=-1, eps=1e-8)
            
            # Compute Cosine Similarity against ONLY the Unseen Prompts -> Shape: (B, U)
            logits = torch.matmul(graph_feat, text_embeddings.T) 
            
            _, top5_indices = logits.topk(min(5, len(unseen_class_names)), dim=-1, largest=True, sorted=True)
            
            for j in range(B_filtered):
                true_local_label = local_labels[j]
                total_samples += 1
                
                if top5_indices[j][0].item() == true_local_label:
                    top1_correct += 1
                if true_local_label in top5_indices[j].tolist():
                    top5_correct += 1

            if total_samples % 1000 == 0 and total_samples > 0:
                print(f"    - Processed {total_samples} unseen instances")

    if total_samples > 0:
        top1_acc = (top1_correct / total_samples) * 100.0
        top5_acc = (top5_correct / total_samples) * 100.0
    else:
        top1_acc = top5_acc = 0.0

    print("\n" + "="*65)
    print(f" STANDARD ZSL RESULTS: {args.split.replace('_', '/')} SPLIT")
    print("="*65)
    print(f"Total Evaluated Unseen Instances: {total_samples}")
    print(f"Search Space: {len(unseen_class_names)} Classes")
    print(f"\n  Top-1 Zero-Shot Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Zero-Shot Accuracy: {top5_acc:.2f}%")
    print("="*65)

if __name__ == "__main__":
    main()