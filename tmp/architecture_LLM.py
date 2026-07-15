import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel, SiglipVisionConfig
import math

# Assumption: You will place your standard ST-GCN implementation in stgcn_backbone.py
# from stgcn_backbone import STGCN

class SiglipVisionEncoder(nn.Module):
    """
    Contextual Vision Stream: Freezes a pre-trained SigLIP vision model.
    Extracts dense, patch-level semantics.
    """
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        super().__init__()
        self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        
        # Freeze the ViT to prevent catastrophic forgetting and save VRAM
        for param in self.vision_model.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.vision_model.config.hidden_size # Usually 768 for base
        self.num_patches = (self.vision_model.config.image_size // self.vision_model.config.patch_size) ** 2

    def forward(self, pixel_values):
        # pixel_values: (B, T, C, H, W)
        B, T, C, H, W = pixel_values.shape
        pixel_values = pixel_values.view(B * T, C, H, W)
        
        with torch.no_grad():
            outputs = self.vision_model(pixel_values)
            
        # Extract patch tokens (ignoring the global pooled output for dense grounding)
        patch_embeds = outputs.last_hidden_state # (B*T, num_patches, hidden_size)
        return patch_embeds.view(B, T, self.num_patches, self.hidden_size)


class TemporalQueryPooler(nn.Module):
    """
    Task 2.1: Modality Synchronization.
    Uses learnable query tokens to compress the high-frequency [t-dt, t+dt] 
    kinematic window into a fixed continuous latent vector representing frame T.
    """
    def __init__(self, stgcn_dim=256, embed_dim=768, num_queries=1):
        super().__init__()
        self.num_queries = num_queries
        
        # FIXED: Removed the extra dimension. Shape is now (1, num_queries, embed_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        self.kv_proj = nn.Linear(stgcn_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, stgcn_features):
        # stgcn_features: (B, T, Max_Subjects, Window_Size, stgcn_dim)
        B, T, S, W, D = stgcn_features.shape
        
        # Flatten Subjects and Window to act as the KV sequence for the pooler
        kv = stgcn_features.view(B * T, S * W, D)
        kv = self.kv_proj(kv)
        
        queries = self.query_tokens.expand(B * T, -1, -1)
        
        # Pooler cross-attention
        pooled_out, _ = self.cross_attn(queries, kv, kv)
        pooled_out = self.norm(pooled_out)
        
        return pooled_out.view(B, T, self.num_queries, -1)


class RoPE2D(nn.Module):
    """
    Task 2.2: 2D Rotary Positional Embeddings.
    Preserves 2D geometric structure within the 1D flattened visual tokens.
    """
    def __init__(self, dim, grid_h=14, grid_w=14):
        super().__init__()
        self.dim = dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 4).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: (B*T, num_patches, dim)
        seq_len = x.shape[1]
        
        pos_y = torch.arange(self.grid_h, device=x.device).repeat_interleave(self.grid_w)
        pos_x = torch.arange(self.grid_w, device=x.device).repeat(self.grid_h)
        
        sin_inp_y = torch.einsum("i,j->ij", pos_y.float(), self.inv_freq)
        sin_inp_x = torch.einsum("i,j->ij", pos_x.float(), self.inv_freq)
        
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        
        emb = torch.cat((emb_y, emb_x), dim=-1).unsqueeze(0) # (1, num_patches, dim)
        return x + emb # Simple additive RoPE injection for keys/values


class CrossAttentionAdaptor(nn.Module):
    """
    Task 2.3 & 2.2: The core Fusion Module.
    Attends pooled kinematic graph queries (anchored with native 2D coordinates) 
    to RoPE-injected visual key/values.
    """
    def __init__(self, vis_dim=768, graph_dim=768, llm_dim=4096):
        super().__init__()
        # Concatenating the native 2D SDK Anchor (colorX, colorY) to the graph token
        self.q_proj = nn.Linear(graph_dim + 2, vis_dim) 
        
        self.rope2d = RoPE2D(vis_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=vis_dim, num_heads=8, batch_first=True)
        
        # The final projection to the LLM (Qwen 35B) latent space
        self.llm_proj = nn.Sequential(
            nn.Linear(vis_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, visual_tokens, pooled_graph_tokens, spatial_anchors):
        # visual_tokens: (B, T, num_patches, vis_dim)
        # pooled_graph_tokens: (B, T, num_queries, graph_dim)
        # spatial_anchors: (B, T, Max_Subjects, 2)
        
        B, T, P, V_D = visual_tokens.shape
        _, _, Q, G_D = pooled_graph_tokens.shape
        
        v_flat = visual_tokens.view(B * T, P, V_D)
        
        # Inject RoPE-2D into the visual tokens (Keys/Values)
        v_flat = self.rope2d(v_flat)
        
        # Average anchors across subjects just for the macroscopic query anchor
        # In more advanced variants, we'd attend to each subject separately
        avg_anchor = spatial_anchors.mean(dim=2) # (B, T, 2)
        avg_anchor = avg_anchor.unsqueeze(2).expand(-1, -1, Q, -1) # (B, T, Q, 2)
        
        # Concat graph tokens with explicit [colorX, colorY]
        q_concat = torch.cat([pooled_graph_tokens, avg_anchor], dim=-1) # (B, T, Q, graph_dim + 2)
        q_flat = self.q_proj(q_concat.view(B * T, Q, G_D + 2))
        
        # Grounding Cross-Attention
        grounded_tokens, _ = self.cross_attn(q_flat, v_flat, v_flat)
        
        # Extract features for InfoNCE contrastive loss (Task 2.3)
        contrastive_visual = v_flat.mean(dim=1) # (B*T, vis_dim)
        contrastive_graph = grounded_tokens.mean(dim=1) # (B*T, vis_dim)
        
        # Project to target LLM dimension
        llm_aligned_tokens = self.llm_proj(grounded_tokens)
        llm_aligned_tokens = llm_aligned_tokens.view(B, T * Q, -1)
        
        return llm_aligned_tokens, contrastive_visual, contrastive_graph


class PGMTDualStream(nn.Module):
    """
    Master Framework class wrapping all Phase 1 Modules.
    """
    def __init__(self, llm_dim=4096): # 4096 is placeholder for Qwen base dim
        super().__init__()
        self.vision_encoder = SiglipVisionEncoder()
        
        # Assume STGCN returns latent dim 256
        # self.kinematic_encoder = STGCN(...) 
        
        self.temporal_pooler = TemporalQueryPooler(stgcn_dim=256, embed_dim=self.vision_encoder.hidden_size)
        self.fusion_adaptor = CrossAttentionAdaptor(
            vis_dim=self.vision_encoder.hidden_size, 
            graph_dim=self.vision_encoder.hidden_size, 
            llm_dim=llm_dim
        )

    def forward(self, batch):
        """
        Expects `batch` dictionary directly from `NTURGBD_AlignedDataset`
        """
        # 1. Vision Stream
        vis_tokens = self.vision_encoder(batch["visual_frames"])
        
        # 2. Kinematic Stream
        # stgcn_features = self.kinematic_encoder(batch["kinematic_windows"])
        # Placeholder for ST-GCN output (B, T, Max_Subjects, Window_Size, 256)
        B, T, S, W, _, _ = batch["kinematic_windows"].shape
        stgcn_features = torch.randn(B, T, S, W, 256, device=vis_tokens.device) 
        
        # 3. Temporal Pooling
        pooled_graph = self.temporal_pooler(stgcn_features)
        
        # 4. RoPE-2D Cross-Attention & Alignment
        llm_tokens, c_vis, c_graph = self.fusion_adaptor(
            vis_tokens, 
            pooled_graph, 
            batch["spatial_anchors"]
        )
        
        return {
            "llm_aligned_tokens": llm_tokens, # Route this directly to Qwen!
            "info_nce_visual": c_vis,         # Route to contrastive loss func
            "info_nce_graph": c_graph         # Route to contrastive loss func
        }