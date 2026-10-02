import math
import logging
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# =====================================================================
# MODULE 1: MULTI-STREAM HYPERGRAPH TRANSFORMER (MS-HyperTR)
# =====================================================================
class MS_HyperTR(nn.Module):
    """
    Processes Joints, Bones, and Motion streams simultaneously using FlashAttention.
    """
    def __init__(self, num_nodes=25, in_dim=3, hidden_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        
        self.joint_embed = nn.Linear(in_dim, hidden_dim)
        self.bone_embed = nn.Linear(in_dim, hidden_dim)
        self.motion_embed = nn.Linear(in_dim, hidden_dim)
        
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, 1, 1, num_nodes, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=0.1, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers, 
            enable_nested_tensor=False
        )

    def forward(self, physics_streams):
        B, T, Streams, W, S, V, C = physics_streams.shape
        
        joints = physics_streams[:, :, 0]
        bones = physics_streams[:, :, 1]
        motion = physics_streams[:, :, 2]
        
        e_joints = self.joint_embed(joints)
        e_bones = self.bone_embed(bones)
        e_motion = self.motion_embed(motion)
        
        fused_physics = e_joints + e_bones + e_motion
        fused_physics = fused_physics + self.spatial_pos
        
        seq_len = W * S * V
        flat_seq = fused_physics.view(B * T, seq_len, -1)
        
        out_seq = self.transformer(flat_seq)
        out_seq = out_seq.view(B, T, W, S, V, -1).mean(dim=4) 
        
        return out_seq.permute(0, 1, 3, 2, 4).contiguous()


# =====================================================================
# MODULE 2: POOLING & CROSS-ATTENTION ADAPTORS
# =====================================================================
class TemporalQueryPooler(nn.Module):
    def __init__(self, input_dim=256, embed_dim=1024, num_queries=4):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.kv_proj = nn.Linear(input_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ms_hypertr_features):
        B, T, S, W, D = ms_hypertr_features.shape
        kv = ms_hypertr_features.view(B * T, S * W, D)
        kv = self.kv_proj(kv)
        
        queries = self.query_tokens.expand(B * T, -1, -1)
        pooled_out, _ = self.cross_attn(queries, kv, kv)
        return self.norm(pooled_out).view(B, T, self.num_queries, -1)


class DirectCrossAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(embed_dim)

    def forward(self, query, key, value):
        attn_scores = torch.bmm(query, key.transpose(1, 2)) * self.scale 
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.bmm(attn_weights, value) 
        return out


# =====================================================================
# MODULE 3: MASTER SPLIT-TRAINING WRAPPER
# =====================================================================
class PGMTSplitStream(nn.Module):
    """
    Tier 3 Retraining Wrapper.
    Replaces the Vision Encoder with pre-extracted RAM tensors.
    Trains strictly the MS-HyperTR and Contrastive Heads.
    """
    def __init__(self, vis_dim=1024, stage1_pretraining=True, llm_dim=4096): 
        super().__init__()
        self.stage1_pretraining = stage1_pretraining
        
        # Vision Encoder completely removed here to save VRAM and execution time.
        self.kinematic_encoder = MS_HyperTR(hidden_dim=256) 
        self.temporal_pooler = TemporalQueryPooler(input_dim=256, embed_dim=vis_dim, num_queries=4)
        
        # -----------------------------------------------------
        # DUAL-LOSS PATHWAY A: Geometric Grounding (Fused Space)
        # -----------------------------------------------------
        self.q_proj_geo = nn.Linear(vis_dim, vis_dim) 
        self.k_proj_geo = nn.Linear(vis_dim, vis_dim)
        self.norm_q_geo = nn.LayerNorm(vis_dim)
        self.norm_kv_geo = nn.LayerNorm(vis_dim)
        self.cross_attn_geo = DirectCrossAttention(embed_dim=vis_dim)
        self.logit_scale_geo = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        
        # -----------------------------------------------------
        # DUAL-LOSS PATHWAY B: Semantic Anchor (Pure SigLIP Space)
        # -----------------------------------------------------
        self.q_proj_sem = nn.Linear(vis_dim, vis_dim)
        self.norm_q_sem = nn.LayerNorm(vis_dim)
        self.logit_scale_sem = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        
        if not self.stage1_pretraining:
            self.llm_proj = nn.Linear(vis_dim, llm_dim)

    def forward(self, batch):
        # Read pre-computed tensors directly from the fast dataloader batch
        vis_tokens_fused = batch["geo_fused"]             # (B, T, 576, 1024)
        vis_tokens_siglip_global = batch["sem_global"]    # (B, T, 1024)
        B, T = vis_tokens_fused.shape[:2]
        
        # 1. Kinematic Forward (3 streams into 4 queries)
        ms_features = self.kinematic_encoder(batch["kinematic_streams"]) 
        Q_graph = self.temporal_pooler(ms_features) 
        
        # ==========================================================
        # PATHWAY A: Geometric Grounding
        # ==========================================================
        v_flat_geo = vis_tokens_fused.view(B * T, 576, -1)
        v_flat_norm_geo = self.norm_kv_geo(v_flat_geo)
        k_flat_geo = self.k_proj_geo(v_flat_norm_geo)
        
        q_flat_geo = self.norm_q_geo(self.q_proj_geo(Q_graph.view(B * T, 4, -1)))
        
        # ==========================================================
        # PATHWAY B: Semantic Text Anchor (Global)
        # ==========================================================
        v_flat_sem_global = vis_tokens_siglip_global.view(B * T, -1)
        q_flat_sem = self.norm_q_sem(self.q_proj_sem(Q_graph.view(B * T, 4, -1)))

        # ==========================================================
        # Anchor Target Extraction with NaN Guardrail
        # ==========================================================
        flat_anchors = batch["spatial_anchors"][:, :, 0, :].contiguous().view(B * T, 2)
        flat_anchors = torch.nan_to_num(flat_anchors, nan=0.5, posinf=0.5, neginf=0.5)
        patch_x = (flat_anchors[:, 0] * 24).clamp(0, 23).long()
        patch_y = (flat_anchors[:, 1] * 24).clamp(0, 23).long()
        patch_idx = ((patch_y * 24) + patch_x).clamp(0, 575)
        batch_indices = torch.arange(B * T, device=v_flat_geo.device)
        
        target_visual_key_geo = k_flat_geo[batch_indices, patch_idx, :] 
        contrastive_graph_geo = q_flat_geo.mean(dim=1) 
        
        target_visual_key_sem = v_flat_sem_global 
        contrastive_graph_sem = q_flat_sem.mean(dim=1)
        
        # FP32 & Zero-Division Safety
        target_visual_key_geo = F.normalize(target_visual_key_geo.float(), p=2, dim=-1, eps=1e-8)
        contrastive_graph_geo = F.normalize(contrastive_graph_geo.float(), p=2, dim=-1, eps=1e-8)
        
        target_visual_key_sem = F.normalize(target_visual_key_sem.float(), p=2, dim=-1, eps=1e-8)
        contrastive_graph_sem = F.normalize(contrastive_graph_sem.float(), p=2, dim=-1, eps=1e-8)
        
        safe_logit_scale_geo = torch.clamp(self.logit_scale_geo, max=4.6052).exp()
        safe_logit_scale_sem = torch.clamp(self.logit_scale_sem, max=4.6052).exp()
        
        llm_aligned_tokens = None
        if not self.stage1_pretraining:
            grounded_tokens_geo = self.cross_attn_geo(query=q_flat_geo, key=k_flat_geo, value=v_flat_norm_geo)
            final_fused_tokens = grounded_tokens_geo + q_flat_sem
            llm_aligned_tokens = self.llm_proj(final_fused_tokens).view(B, T * 4, -1)
        
        return {
            "llm_aligned_tokens": llm_aligned_tokens,
            "info_nce_visual_geo": target_visual_key_geo,
            "info_nce_graph_geo": contrastive_graph_geo,
            "logit_scale_geo": safe_logit_scale_geo,
            "info_nce_visual_sem": target_visual_key_sem,
            "info_nce_graph_sem": contrastive_graph_sem,
            "logit_scale_sem": safe_logit_scale_sem
        }

# =====================================================================
# VERIFICATION BLOCK
# =====================================================================
if __name__ == "__main__":
    print("[*] Testing Tier 3 Split Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PGMTSplitStream(vis_dim=1024, stage1_pretraining=True).to(device)
    
    # Simulate pre-extracted RAM tensors from dataset_split.py
    dummy_batch = {
        "geo_fused": torch.randn(2, 3, 576, 1024).to(device),
        "sem_global": torch.randn(2, 3, 1024).to(device),
        "kinematic_streams": torch.randn(2, 3, 3, 30, 2, 25, 3).to(device),
        "spatial_anchors": torch.rand(2, 3, 2, 2).to(device)
    }
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type if device.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
            outputs = model(dummy_batch)
            
    print("\nOutputs Validated:")
    for key, val in outputs.items():
        if val is not None:
            print(f"  - {key}: {val.shape}")
