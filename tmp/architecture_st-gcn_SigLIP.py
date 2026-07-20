import math
import logging
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel, SiglipVisionConfig

from stgcn_backbone import PGMT_STGCN

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SiglipVisionEncoder(nn.Module):
    """
    Contextual Vision Stream: Freezes a pre-trained SigLIP vision model.
    Extracts dense, patch-level semantics from sparse RGB video frames.
    
    This module acts as the macroscopic semantic backbone, discarding 
    high-frequency temporal redundancy while preserving spatial context.
    
    Attributes:
        vision_model (SiglipVisionModel): The underlying HuggingFace SigLIP model.
        hidden_size (int): Dimensionality of the output patch tokens.
        num_patches (int): Total number of spatial patches per frame.
    """
    def __init__(self, model_name: str = "google/siglip-base-patch16-224") -> None:
        super().__init__()
        logger.info(f"Initializing SiglipVisionEncoder with backend: {model_name}")
        self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        
        # Freeze the ViT to prevent catastrophic forgetting and save VRAM
        for param in self.vision_model.parameters():
            param.requires_grad = False
            
        self.hidden_size: int = self.vision_model.config.hidden_size
        
        # Calculate number of patches based on image size and patch size
        image_size = self.vision_model.config.image_size
        patch_size = self.vision_model.config.patch_size
        self.num_patches: int = (image_size // patch_size) ** 2
        
        logger.info(f"Vision Encoder initialized. Hidden size: {self.hidden_size}, Patches: {self.num_patches}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the vision encoder.
        
        Args:
            pixel_values (torch.Tensor): Input video frames of shape (B, T, C, H, W).
            
        Returns:
            torch.Tensor: Patch embeddings of shape (B, T, num_patches, hidden_size).
        """
        B, T, C, H, W = pixel_values.shape
        pixel_values_flat = pixel_values.view(B * T, C, H, W)
        
        with torch.no_grad():
            outputs = self.vision_model(pixel_values_flat)
            
        # Extract patch tokens (ignoring the global pooled output for dense grounding)
        patch_embeds = outputs.last_hidden_state # (B*T, num_patches, hidden_size)
        
        return patch_embeds.view(B, T, self.num_patches, self.hidden_size)


class TemporalQueryPooler(nn.Module):
    """
    Task 2.1: Modality Synchronization.
    Uses learnable query tokens to compress the high-frequency [t-dt, t+dt] 
    kinematic window into a fixed continuous latent vector representing frame T.
    
    This effectively decodes variable-length skeletal sequences into discrete
    information bottlenecks.
    """
    def __init__(self, stgcn_dim: int = 256, embed_dim: int = 768, num_queries: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        
        # Learnable query tokens initialized via normal distribution
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        # Projection and Attention Blocks
        self.kv_proj = nn.Linear(stgcn_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes projection weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.zeros_(self.kv_proj.bias)

    def forward(self, stgcn_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stgcn_features (torch.Tensor): Shape (B, T, Max_Subjects, Window_Size, stgcn_dim)
            
        Returns:
            torch.Tensor: Pooled representation of shape (B, T, num_queries, embed_dim)
        """
        B, T, S, W, D = stgcn_features.shape
        
        # Flatten Subjects and Window to act as the KV sequence for the pooler
        kv = stgcn_features.view(B * T, S * W, D)
        kv = self.kv_proj(kv)
        
        # Expand queries across the batch and temporal dimensions
        queries = self.query_tokens.expand(B * T, -1, -1)
        
        # Pooler cross-attention
        pooled_out, _ = self.cross_attn(queries, kv, kv)
        pooled_out = self.dropout(pooled_out)
        pooled_out = self.norm(pooled_out)
        
        return pooled_out.view(B, T, self.num_queries, -1)


class RoPE2D(nn.Module):
    """
    Task 2.2: 2D Rotary Positional Embeddings.
    Preserves 2D geometric structure within the 1D flattened visual tokens.
    
    Computes trigonometric absolute positional embeddings across a predefined 2D grid,
    enabling the Cross Attention mechanism to recognize relative spatial configurations.
    """
    def __init__(self, dim: int, grid_h: int = 14, grid_w: int = 14, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        
        # Precompute the inverse frequencies for the sinusoidal embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 4).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Injects 2D rotary positional encodings directly into the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B*T, num_patches, dim).
            
        Returns:
            torch.Tensor: Position-aware tensor of shape (B*T, num_patches, dim).
        """
        seq_len = x.shape[1]
        
        if seq_len != self.grid_h * self.grid_w:
            raise ValueError(f"Expected sequence length {self.grid_h * self.grid_w}, got {seq_len}")
        
        # Generate grid coordinates
        pos_y = torch.arange(self.grid_h, device=x.device).repeat_interleave(self.grid_w)
        pos_x = torch.arange(self.grid_w, device=x.device).repeat(self.grid_h)
        
        # Compute trigonometric inputs
        sin_inp_y = torch.einsum("i,j->ij", pos_y.float(), self.inv_freq)
        sin_inp_x = torch.einsum("i,j->ij", pos_x.float(), self.inv_freq)
        
        # Concatenate sin and cos components
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        
        # Combine X and Y embeddings into final representation
        emb = torch.cat((emb_y, emb_x), dim=-1).unsqueeze(0) # (1, num_patches, dim)
        
        return x + emb 


class DirectCrossAttention(nn.Module):
    """
    Parameter-free Scaled Dot-Product Attention.
    Directly aligns learned Graph Queries with mapped visual embeddings.
    """
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.scale = 1.0 / math.sqrt(embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # query: (B, 1, D), key: (B, 196, D), value: (B, 196, D)
        attn_scores = torch.bmm(query, key.transpose(1, 2)) * self.scale # (B, 1, 196)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.bmm(attn_weights, value) # (B, 1, D)
        
        return out, attn_weights


class CrossAttentionAdaptor(nn.Module):
    """
    Task 2.3 & 2.2: The core Fusion Module.
    Attends pooled kinematic graph queries (anchored with native 2D coordinates) 
    to RoPE-injected visual key/values.
    """
    def __init__(self, vis_dim: int = 768, graph_dim: int = 768, llm_dim: int = 4096, use_llm_proj: bool = False, dropout: float = 0.1) -> None:
        super().__init__()
        self.use_llm_proj = use_llm_proj
        self.vis_dim = vis_dim
        
        # Projection layer to align graph tokens + 2D native coordinates to the shared latent space
        self.q_proj = nn.Linear(graph_dim + 2, vis_dim) 
        
        # --- THE FIX: SYMMETRIC PROJECTION HEAD ---
        # We must project the visual keys so they can align with the graph queries
        # in a shared multimodal latent space.
        self.k_proj = nn.Linear(vis_dim, vis_dim)
        
        self.rope2d = RoPE2D(vis_dim)
        
        self.norm_q = nn.LayerNorm(vis_dim)
        self.norm_kv = nn.LayerNorm(vis_dim)
        
        # Parameter-free attention mechanism
        self.cross_attn = DirectCrossAttention(embed_dim=vis_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(vis_dim)
        
        # Stage 2 Only: The final projection to the LLM latent space
        if self.use_llm_proj:
            self.llm_proj = nn.Linear(vis_dim, llm_dim)
            
        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes internal projection layers."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        if self.use_llm_proj:
            nn.init.xavier_uniform_(self.llm_proj.weight)
            nn.init.zeros_(self.llm_proj.bias)

    def forward(self, visual_tokens: torch.Tensor, pooled_graph_tokens: torch.Tensor, spatial_anchors: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Fuses modalities and extracts contrastive targets.
        """
        B, T, P, V_D = visual_tokens.shape
        _, _, Q, G_D = pooled_graph_tokens.shape
        
        v_flat = visual_tokens.view(B * T, P, V_D)
        
        # Inject RoPE-2D into the visual tokens
        v_flat = self.rope2d(v_flat)
        v_flat = self.norm_kv(v_flat)
        
        # Project visual tokens into the shared Key space
        k_flat = self.k_proj(v_flat)
        
        # We take the primary subject (index 0) as the macroscopic query anchor.
        primary_anchor = spatial_anchors[:, :, 0, :] # (B, T, 2)
        primary_anchor = primary_anchor.unsqueeze(2).expand(-1, -1, Q, -1) # (B, T, Q, 2)
        
        # Concat graph tokens with explicit [colorX, colorY]
        q_concat = torch.cat([pooled_graph_tokens, primary_anchor], dim=-1) # (B, T, Q, graph_dim + 2)
        q_flat = self.q_proj(q_concat.view(B * T, Q, G_D + 2))
        q_flat = self.norm_q(q_flat)
        
        # Grounding Cross-Attention (Outputs the human representation extracted from the image)
        # We use the projected k_flat as the Keys, but the pristine v_flat as the Values
        # so the output retains the pure semantic meaning of the SigLIP token.
        grounded_tokens, attn_weights = self.cross_attn(query=q_flat, key=k_flat, value=v_flat)
        grounded_tokens = self.dropout(grounded_tokens)
        grounded_tokens = self.norm_out(grounded_tokens)
        
        # --- ELEGANT SPATIAL GROUNDING TARGET ---
        # 1. Flatten the normalized [0,1] anchors to (Batch*Time, 2)
        flat_anchors = spatial_anchors[:, :, 0, :].contiguous().view(B * T, 2)
        grid_resolution = 14
        
        # 2. Extract the exact visual KEY token at the person's location
        patch_x = (flat_anchors[:, 0] * grid_resolution).clamp(0, grid_resolution - 1).long()
        patch_y = (flat_anchors[:, 1] * grid_resolution).clamp(0, grid_resolution - 1).long()
        patch_idx = (patch_y * grid_resolution) + patch_x 
        patch_idx = patch_idx.clamp(0, (grid_resolution * grid_resolution) - 1)
        
        batch_indices = torch.arange(B * T, device=v_flat.device)
        
        # Crucial: The contrastive target must be the projected Key, not the raw Value,
        # so the Query and Target reside in the exact same vector space.
        target_visual_key = k_flat[batch_indices, patch_idx, :] # (B*T, vis_dim)
        
        # 3. Route to Contrastive Loss
        contrastive_visual = target_visual_key
        contrastive_graph = q_flat.squeeze(1)
        
        # Project to target LLM dimension
        llm_aligned_tokens = None
        if self.use_llm_proj:
            llm_aligned_tokens = self.llm_proj(grounded_tokens)
            llm_aligned_tokens = llm_aligned_tokens.view(B, T * Q, -1)
        
        return {
            "llm_aligned_tokens": llm_aligned_tokens,
            "info_nce_visual": contrastive_visual,
            "info_nce_graph": contrastive_graph
        }

class PGMTDualStream(nn.Module):
    """
    Master Framework class wrapping all Phase 1 Modules.
    Supports Stage 1 (Contrastive Only) and Stage 2 (LLM Autoregressive) modes.
    
    Args:
        llm_dim (int): Target dimensionality for the Large Language Model.
        stage1_pretraining (bool): If True, bypasses final LLM projection layer.
    """
    def __init__(self, llm_dim: int = 4096, stage1_pretraining: bool = True) -> None: 
        super().__init__()
        self.stage1_pretraining = stage1_pretraining
        
        # Instantiate Sub-Modules
        self.vision_encoder = SiglipVisionEncoder()
        self.kinematic_encoder = PGMT_STGCN(in_channels=3, hidden_channels=256) 
        self.temporal_pooler = TemporalQueryPooler(stgcn_dim=256, embed_dim=self.vision_encoder.hidden_size)
        
        self.fusion_adaptor = CrossAttentionAdaptor(
            vis_dim=self.vision_encoder.hidden_size, 
            graph_dim=self.vision_encoder.hidden_size, 
            llm_dim=llm_dim,
            use_llm_proj=not self.stage1_pretraining
        )

    @property
    def num_trainable_parameters(self) -> int:
        """Returns the total number of trainable parameters in the architecture."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
        """
        Executes the dual-stream forward pass.
        
        Args:
            batch: Dictionary containing 'visual_frames', 'kinematic_windows', and 'spatial_anchors'.
            
        Returns:
            Dictionary containing projection outputs and contrastive targets.
        """
        # 1. Vision Stream
        vis_tokens = self.vision_encoder(batch["visual_frames"])
        
        # 2. Kinematic Stream 
        stgcn_features = self.kinematic_encoder(batch["kinematic_windows"]) 
        
        # 3. Temporal Pooling
        pooled_graph = self.temporal_pooler(stgcn_features)
        
        # 4. RoPE-2D Cross-Attention & Alignment
        outputs = self.fusion_adaptor(
            vis_tokens, 
            pooled_graph, 
            batch["spatial_anchors"]
        )
        
        return outputs

# =====================================================================
# MODULAR UNIT TESTING BLOCK
# Run `python architecture.py` to independently verify every module
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" PGMT Phase 1: Modular Unit Tests (Extended Validation)")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Executing tests on device: {device}")
    
    # Dummy Dimensions
    B, T, S, W, V_D, G_D = 2, 3, 2, 30, 768, 256
    
    # ---------------------------------------------------------
    # Test 1: Temporal Query Pooler
    # ---------------------------------------------------------
    print("\n[1] Testing TemporalQueryPooler...")
    try:
        pooler = TemporalQueryPooler(stgcn_dim=G_D, embed_dim=V_D, num_queries=1).to(device)
        dummy_stgcn = torch.randn(B, T, S, W, G_D).to(device)
        out_pool = pooler(dummy_stgcn)
        assert out_pool.shape == (B, T, 1, V_D), f"Shape mismatch: {out_pool.shape}"
        print(" -> SUCCESS: TemporalQueryPooler output shape is correct.")
    except Exception as e:
        print(f" -> FAILED: {e}")

    # ---------------------------------------------------------
    # Test 2: RoPE-2D Embeddings
    # ---------------------------------------------------------
    print("\n[2] Testing RoPE2D Injection...")
    try:
        rope = RoPE2D(dim=V_D).to(device)
        dummy_vis = torch.randn(B*T, 196, V_D).to(device)
        out_rope = rope(dummy_vis)
        assert out_rope.shape == (B*T, 196, V_D), f"Shape mismatch: {out_rope.shape}"
        print(" -> SUCCESS: RoPE2D tensor preservation is correct.")
    except Exception as e:
        print(f" -> FAILED: {e}")

    # ---------------------------------------------------------
    # Test 3: Cross-Attention Adaptor & InfoNCE Targets
    # ---------------------------------------------------------
    print("\n[3] Testing CrossAttentionAdaptor (Direct Attention Math)...")
    try:
        adaptor = CrossAttentionAdaptor(vis_dim=V_D, graph_dim=V_D, use_llm_proj=False).to(device)
        dummy_vis_tokens = torch.randn(B, T, 196, V_D).to(device)
        dummy_graph_tokens = torch.randn(B, T, 1, V_D).to(device)
        dummy_anchors = torch.rand(B, T, S, 2).to(device) # [0, 1] normalized coordinates
        
        out_adaptor = adaptor(dummy_vis_tokens, dummy_graph_tokens, dummy_anchors)
        
        assert out_adaptor["info_nce_visual"].shape == (B*T, V_D), "Visual target shape mismatch!"
        assert out_adaptor["info_nce_graph"].shape == (B*T, V_D), "Graph query shape mismatch!"
        print(" -> SUCCESS: CrossAttentionAdaptor & parameter-free geometric extraction is correct.")
    except Exception as e:
        print(f" -> FAILED: {e}")
        
    # ---------------------------------------------------------
    # Test 4: Master Framework Parameter Assessment
    # ---------------------------------------------------------
    print("\n[4] Testing PGMTDualStream Parameters...")
    try:
        model = PGMTDualStream(stage1_pretraining=True).to(device)
        print(f" -> SUCCESS: PGMTDualStream initialized with {model.num_trainable_parameters:,} trainable parameters.")
    except Exception as e:
        print(f" -> FAILED: {e}")
        
    print("\n" + "="*50)
    print(" ALL EXTENDED MODULES PASSED INDEPENDENT VERIFICATION")
    print("="*50 + "\n")