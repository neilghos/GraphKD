import torch
import torch.nn as nn
import timm

class DualReconstructionStudent(nn.Module):
    """
    Dual-Reconstruction Student for Multi-Teacher Knowledge Distillation (Option B).
    
    Architecture:
      - Shared Core: ConvNeXt-Femto (native 384d pooled core).
      - ConvNeXt Specialist: 640d Depthwise Conv block.
        Reconstructed ConvNeXt = [spec_conv (640d), shared_core (384d)] = Exactly 1024 dims!
        Head: Linear(1024, 200) -> logits_conv
      - DINOv2 Specialist: 384d Spatial Multi-Head Attention block.
        Reconstructed DINOv2   = [spec_dino (384d), shared_core (384d)] = Exactly 768 dims!
        Head: Linear(768, 200)  -> logits_dino
    """
    def __init__(self, num_classes=200, pretrained=True):
        super().__init__()
        # 1. Shared Trunk: ConvNeXt-Femto (~4.83M params)
        self.trunk = timm.create_model("convnext_femto", pretrained=pretrained, num_classes=0)
        trunk_dim = 384
        self.conv_spec_dim = 1024 - trunk_dim  # 640 dims
        self.dino_spec_dim = 768 - trunk_dim   # 384 dims

        # 2. ConvNeXt Specialist (7x7 Depthwise-Separable Conv -> 640d)
        self.conv_spec = nn.Sequential(
            nn.Conv2d(trunk_dim, trunk_dim, kernel_size=7, padding=3, groups=trunk_dim, bias=False),
            nn.BatchNorm2d(trunk_dim),
            nn.GELU(),
            nn.Conv2d(trunk_dim, self.conv_spec_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.conv_spec_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv_ln = nn.LayerNorm(1024)
        self.head_conv = nn.Linear(1024, num_classes)

        # 3. DINOv2 Specialist (Spatial Multi-Head Self-Attention -> 384d)
        mha_dim = 192
        self.dino_in_proj = nn.Linear(trunk_dim, mha_dim)
        self.dino_mha = nn.MultiheadAttention(embed_dim=mha_dim, num_heads=6, batch_first=True)
        self.dino_mha_ln = nn.LayerNorm(mha_dim)
        self.dino_out_proj = nn.Linear(mha_dim, self.dino_spec_dim)
        self.dino_ln = nn.LayerNorm(768)
        self.head_dino = nn.Linear(768, num_classes)

    def forward(self, x):
        feat_map = self.trunk.forward_features(x)              # [B, 384, 7, 7]
        shared_core = feat_map.mean(dim=[-2, -1])              # [B, 384]

        # ConvNeXt Stream (1024d)
        spec_conv = self.conv_spec(feat_map).flatten(1)        # [B, 640]
        emb_conv = torch.cat([spec_conv, shared_core], dim=-1) # [B, 1024]
        emb_conv_norm = self.conv_ln(emb_conv)
        logits_conv = self.head_conv(emb_conv_norm)            # [B, 200]

        # DINOv2 Stream (768d)
        tokens = feat_map.flatten(2).transpose(1, 2)           # [B, 49, 384]
        h = self.dino_in_proj(tokens)                          # [B, 49, 192]
        attn_out, _ = self.dino_mha(h, h, h)                   # [B, 49, 192]
        h = self.dino_mha_ln(h + attn_out)                     # Residual + LN
        h_pool = h.mean(dim=1)                                 # [B, 192]
        spec_dino = self.dino_out_proj(h_pool)                 # [B, 384]
        emb_dino = torch.cat([spec_dino, shared_core], dim=-1) # [B, 768]
        emb_dino_norm = self.dino_ln(emb_dino)
        logits_dino = self.head_dino(emb_dino_norm)            # [B, 200]

        # Combined Ensemble Logits & Joint 1792d Representation
        logits_ens = 0.5 * logits_conv + 0.5 * logits_dino
        emb_joint = torch.cat([emb_conv, emb_dino], dim=-1)     # [B, 1792]

        return logits_conv, logits_dino, logits_ens, emb_conv, emb_dino, emb_joint
