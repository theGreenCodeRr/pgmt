import torch
import traceback
from dataset_ntu import NTURGBD_AlignedDataset
from architecture import PGMTDualStream

def generate_mock_batch():
    """Generates dummy tensors mimicking the NTU dataset output if real data isn't found."""
    print("[!] Generating mock data batch for architecture validation...")
    B, T, Max_S, W, Nodes = 1, 3, 2, 30, 25
    return {
        "visual_frames": torch.randn(B, T, 3, 224, 224),
        # FIXED: Ordered dimensions to (Batch, Time, Window, Subjects, Nodes, Coordinates)
        "kinematic_windows": torch.randn(B, T, W, Max_S, Nodes, 3),
        "spatial_anchors": torch.rand(B, T, Max_S, 2) * 224 # Simulated pixel coordinates
    }

def main():
    print("="*50)
    print(" PGMT Phase 1: Architecture Forward Pass Test (Stage 1)")
    print("="*50)

    # 1. Initialize Dataset
    try:
        dataset = NTURGBD_AlignedDataset(data_root="E:\\datasets\\ntu_rgbd")
        if len(dataset) > 0:
            print(f"[*] Successfully loaded {len(dataset)} samples.")
            # Add batch dimension to simulate a DataLoader
            batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                     for k, v in dataset[0].items() if k != "sample_id"}
        else:
            print("[!] Dataset initialized but found 0 aligned samples. Check E:\\datasets\\ntu_rgbd")
            batch = generate_mock_batch()
    except Exception as e:
        print(f"[!] Dataset loading failed: {e}")
        traceback.print_exc()
        batch = generate_mock_batch()

    # 2. Initialize the PGMT Dual-Stream Architecture
    print("\n[*] Initializing PGMTDualStream (Stage 1 Pretraining Mode)...")
    try:
        # Note: stage1_pretraining=True is the default now, disabling LLM projection
        model = PGMTDualStream(stage1_pretraining=True)
        model.eval() # Set to evaluation mode for testing
        print("[*] Model successfully initialized. LLM projection bypassed for VRAM efficiency.")
    except Exception as e:
        print(f"[!] Model initialization failed: {e}")
        return

    # 3. Perform Forward Pass
    print("\n[*] Pushing tensors through the network...")
    try:
        outputs = model(batch)
        print("\n" + "="*50)
        print(" FORWARD PASS SUCCESSFUL - STAGE 1 OUTPUT TENSOR SHAPES")
        print("="*50)
        
        # Verify our target Qwen MoE projection is bypassed
        print(f"1. LLM Aligned Tokens:  {outputs['llm_aligned_tokens']} -> (Expected None during Stage 1)")
        
        # Verify our Contrastive Loss intermediate features
        print(f"2. Visual Contrastive:  {outputs['info_nce_visual'].shape} -> Expected (Batch*T, 768)")
        print(f"3. Graph Contrastive:   {outputs['info_nce_graph'].shape} -> Expected (Batch*T, 768)")
        print("="*50)
        print("Ready for InfoNCE Contrastive Training Loop!")
        
    except Exception as e:
        print(f"\n[!] Forward pass failed with matrix mismatch or error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()