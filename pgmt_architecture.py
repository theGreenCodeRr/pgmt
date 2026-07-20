import math
import logging
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================
# MODULE 1: DUAL VISION ENCODER (SigLIP + DINOv2)
# =====================================================================
class DualVisionEncoder(nn.Module):
    """
    Fuses semantic features (SigLIP) with geometric/depth features (DINOv2).
    Maps both to a unified 24x24 spatial grid (576 tokens).
    """
    def __init__(self, output_dim=1024):
        super().__init__()
        logger.info("Loading SigLIP-Large (Semantic) and DINOv2-Large (Geometric)...")
        
        # 1. SigLIP Large (Requires 384x384 input -> outputs 24x24 grid)
        self.siglip = SiglipVisionModel.from_pretrained("google/siglip-large-patch16-384")
        
        # 2. DINOv2 Large (Requires 336x336 input -> outputs 24x24 grid)
        # Using hub.load to fetch the ViT-L/14 model from Facebook
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        
        # Freeze both backbone weights to conserve VRAM
        for param in self.siglip.parameters(): param.requires_grad = False
        for param in self.dinov2.parameters(): param.requires_grad = False
            
        siglip_dim = self.siglip.config.hidden_size # 1024
        dinov2_dim = self.dinov2.embed_dim          # 1024
        
        # MLP Fusion Projector (1024 + 1024 -> 2048 -> 1024)
        self.fusion_projector = nn.Sequential(
            nn.Linear(siglip_dim + dinov2_dim, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, output_dim)
        )

    def forward(self, rgb_siglip, rgb_dinov2):
        # Expected input shape: (B, T, C, H, W)
        B, T, C, H_s, W_s = rgb_siglip.shape
        _, _, _, H_d, W_d = rgb_dinov2.shape
        
        # Flatten batch and time
        rgb_siglip = rgb_siglip.view(B * T, C, H_s, W_s)
        rgb_dinov2 = rgb_dinov2.view(B * T, C, H_d, W_d)
        
        with torch.no_grad():
            # Extract SigLIP Semantics
            siglip_out = self.siglip(rgb_siglip).last_hidden_state # (B*T, 576, 1024)
            
            # Extract DINOv2 Geometry
            dinov2_out = self.dinov2.forward_features(rgb_dinov2)['x_norm_patchtokens'] # (B*T, 576, 1024)
            
        # Concatenate on feature dimension: (B*T, 576, 2048)
        combined_features = torch.cat([siglip_out, dinov2_out], dim=-1)
        
        # Fuse to multimodal Super Token: (B*T, 576, 1024)
        fused_tokens = self.fusion_projector(combined_features)
        
        return fused_tokens.view(B, T, 576, -1)


# =====================================================================
# MODULE 2: MULTI-STREAM HYPERGRAPH TRANSFORMER (MS-HyperTR)
# =====================================================================
class MS_HyperTR(nn.Module):
    """
    Replaces legacy ST-GCN. Processes Joints, Bones, and Motion streams 
    simultaneously using FlashAttention.
    """
    def __init__(self, num_nodes=25, in_dim=3, hidden_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        
        # Independent linear embeddings for the 3 physical streams
        self.joint_embed = nn.Linear(in_dim, hidden_dim)
        self.bone_embed = nn.Linear(in_dim, hidden_dim)
        self.motion_embed = nn.Linear(in_dim, hidden_dim)
        
        # Spatial-Temporal Positional Encodings (Broadcastable to B, T, W, S, V, D)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, 1, 1, num_nodes, hidden_dim))
        
        # Master Transformer Encoder 
        # (Uses FlashAttention natively if torch.amp is enabled on H100)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=0.1, 
            batch_first=True,
            norm_first=True # Better stability for deep networks
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, physics_streams):
        # Input shape from Dataloader: (B, T, 3_streams, Window, Subjects, Nodes, 3_XYZ)
        B, T, Streams, W, S, V, C = physics_streams.shape
        
        joints = physics_streams[:, :, 0]
        bones = physics_streams[:, :, 1]
        motion = physics_streams[:, :, 2]
        
        # Embed each stream
        e_joints = self.joint_embed(joints)
        e_bones = self.bone_embed(bones)
        e_motion = self.motion_embed(motion)
        
        # Fuse streams via summation (Physics Synergies)
        fused_physics = e_joints + e_bones + e_motion # (B, T, W, S, V, hidden_dim)
        
        # Add Node-level geometric positions
        fused_physics = fused_physics + self.spatial_pos
        
        # Flatten sequence for Transformer: Sequence length = (Window * Subjects * Nodes)
        seq_len = W * S * V
        flat_seq = fused_physics.view(B * T, seq_len, -1)
        
        # Apply Transformer (Calculates global hyper-edges between all limbs across time)
        out_seq = self.transformer(flat_seq)
        
        # Reshape and Mean Pool over the V (Nodes) dimension
        out_seq = out_seq.view(B, T, W, S, V, -1).mean(dim=4) # (B, T, W, S, D)
        
        # Permute for Temporal Query Pooler: (B, T, S, W, D)
        return out_seq.permute(0, 1, 3, 2, 4).contiguous()


# =====================================================================
# MODULE 3: POOLING & CROSS-ATTENTION ADAPTORS
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


class RoPE2D(nn.Module):
    """
    2D Rotary Positional Embeddings adapted for the 24x24 Super Token grid.
    """
    def __init__(self, dim, grid_h=24, grid_w=24, base=10000.0):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 4).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        pos_y = torch.arange(self.grid_h, device=x.device).repeat_interleave(self.grid_w)
        pos_x = torch.arange(self.grid_w, device=x.device).repeat(self.grid_h)
        sin_inp_y = torch.einsum("i,j->ij", pos_y.float(), self.inv_freq)
        sin_inp_x = torch.einsum("i,j->ij", pos_x.float(), self.inv_freq)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.cat((emb_y, emb_x), dim=-1).unsqueeze(0)
        return x + emb 


class DirectCrossAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(embed_dim)

    def forward(self, query, key, value):
        attn_scores = torch.bmm(query, key.transpose(1, 2)) * self.scale 
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.bmm(attn_weights, value) 
        return out


class PGMTDualStream(nn.Module):
    """
    Master Wrapper for Phase 1: Dual Vision + MS-HyperTR + Contrastive Alignment
    """
    def __init__(self, vis_dim=1024, stage1_pretraining=True, llm_dim=4096): 
        super().__init__()
        self.stage1_pretraining = stage1_pretraining
        
        self.vision_encoder = DualVisionEncoder(output_dim=vis_dim)
        self.kinematic_encoder = MS_HyperTR(hidden_dim=256) 
        self.temporal_pooler = TemporalQueryPooler(input_dim=256, embed_dim=vis_dim, num_queries=4)
        
        # Alignment & Geometry (Targeting 24x24 grid = 576 tokens)
        self.q_proj = nn.Linear(vis_dim + 2, vis_dim) # +2 for Spine Base Anchor (X, Y)
        self.k_proj = nn.Linear(vis_dim, vis_dim)
        self.rope2d = RoPE2D(vis_dim, grid_h=24, grid_w=24)
        
        self.norm_q = nn.LayerNorm(vis_dim)
        self.norm_kv = nn.LayerNorm(vis_dim)
        self.cross_attn = DirectCrossAttention(embed_dim=vis_dim)
        
        if not self.stage1_pretraining:
            self.llm_proj = nn.Linear(vis_dim, llm_dim)

    def forward(self, batch):
        B, T = batch["visual_frames_siglip"].shape[:2]
        
        # 1. Vision Forward (Dual Resolution)
        vis_tokens = self.vision_encoder(batch["visual_frames_siglip"], batch["visual_frames_dinov2"])
        
        # 2. Kinematic Forward (3-Streams)
        ms_features = self.kinematic_encoder(batch["kinematic_streams"]) 
        Q_graph = self.temporal_pooler(ms_features) # (B, T, 4, vis_dim)
        
        # 3. Geometry Setup
        v_flat = vis_tokens.view(B * T, 576, -1)
        v_flat = self.norm_kv(self.rope2d(v_flat))
        k_flat = self.k_proj(v_flat)
        
        primary_anchor = batch["spatial_anchors"][:, :, 0, :] # Shape: (B, T, 2)
        primary_anchor = primary_anchor.unsqueeze(2).expand(-1, -1, 4, -1) 
        
        # 4. Query Grounding
        q_concat = torch.cat([Q_graph, primary_anchor], dim=-1) 
        q_flat = self.norm_q(self.q_proj(q_concat.view(B * T, 4, -1)))
        
        grounded_tokens = self.cross_attn(query=q_flat, key=k_flat, value=v_flat)
        
        # --- InfoNCE Target Extraction ---
        flat_anchors = batch["spatial_anchors"][:, :, 0, :].contiguous().view(B * T, 2)
        # Using 24x24 grid resolution
        patch_x = (flat_anchors[:, 0] * 24).clamp(0, 23).long()
        patch_y = (flat_anchors[:, 1] * 24).clamp(0, 23).long()
        patch_idx = ((patch_y * 24) + patch_x).clamp(0, 575)
        
        batch_indices = torch.arange(B * T, device=v_flat.device)
        target_visual_key = k_flat[batch_indices, patch_idx, :] # (B*T, vis_dim)
        contrastive_graph = q_flat.mean(dim=1) # Average the 4 queries
        
        llm_aligned_tokens = None
        if not self.stage1_pretraining:
            llm_aligned_tokens = self.llm_proj(grounded_tokens).view(B, T * 4, -1)
        
        return {
            "llm_aligned_tokens": llm_aligned_tokens,
            "info_nce_visual": target_visual_key,
            "info_nce_graph": contrastive_graph
        }