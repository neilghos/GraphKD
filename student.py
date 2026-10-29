import torch
import torch.nn as nn
import timm

class ModularStudent(nn.Module):
    def __init__(self, core_dim=336, sat1_dim=32, sat2_dim=16, num_classes=200, head_type="inductive", pretrained=True):
        super().__init__()
        self.head_type = head_type
        self.core_dim = core_dim
        self.sat1_dim = sat1_dim
        self.sat2_dim = sat2_dim

        # 1. ConvNeXt-Femto Trunk (~4.83M parameters, native 384-dim feature map)
        self.trunk = timm.create_model("convnext_femto", pretrained=pretrained, num_classes=0)
        trunk_dim = 384

        # Homophilic Core Head (Preserves invariant representation)
        self.core_head = nn.Linear(trunk_dim, core_dim)

        # 2. Parallel Satellites for Heterophilic Specialist Regions
        if head_type == "mlp":
            # Plain baseline MLPs
            self.sat1_head = nn.Sequential(nn.Linear(trunk_dim, 64), nn.GELU(), nn.Linear(64, sat1_dim))
            self.sat2_head = nn.Sequential(nn.Linear(trunk_dim, 64), nn.GELU(), nn.Linear(64, sat2_dim))
        elif head_type == "inductive":
            # Operator-Matched Inductive Satellites:
            # Sat 1: 2D 7x7 Depthwise Conv Block (ConvNeXt modern convolutional inductive prior)
            self.sat1_conv = nn.Sequential(
                nn.Conv2d(trunk_dim, trunk_dim, kernel_size=7, padding=3, groups=trunk_dim, bias=False),
                nn.BatchNorm2d(trunk_dim),
                nn.GELU(),
                nn.Conv2d(trunk_dim, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.sat1_proj = nn.Linear(64, sat1_dim)

            # Sat 2: Spatial Multi-Head Self-Attention Block (DINOv2 ViT relational prior)
            self.sat2_in_proj = nn.Linear(trunk_dim, 64)
            self.sat2_mha = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
            self.sat2_ln = nn.LayerNorm(64)
            self.sat2_proj = nn.Linear(64, sat2_dim)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        # 3. Dynamic Gating & Readout
        self.gate = nn.Linear(core_dim, 2)
        total_dim = core_dim + sat1_dim + sat2_dim  # Exactly 384: Femto's native embedding dim
        self.fused_ln = nn.LayerNorm(total_dim)
        self.classifier = nn.Linear(total_dim, num_classes)

    def forward(self, x):
        feat_map = self.trunk.forward_features(x)         # [B, 384, 7, 7]
        feat = feat_map.mean(dim=[-2, -1])                # [B, 384]
        
        # Invariant Core Head
        s_core = self.core_head(feat)                     # [B, core_dim]

        # Specialist Satellites
        if self.head_type == "mlp":
            s_sat1 = self.sat1_head(feat)
            s_sat2 = self.sat2_head(feat)
        else:
            # Sat 1: 7x7 Depthwise Convolution (ConvNeXt Inductive Bias)
            sat1_feat = self.sat1_conv(feat_map).flatten(1)  # [B, 64]
            s_sat1 = self.sat1_proj(sat1_feat)               # [B, sat1_dim]

            # Sat 2: 49 Spatial Tokens All-to-All Self-Attention (DINOv2 Inductive Bias)
            tokens = feat_map.flatten(2).transpose(1, 2)     # [B, 49, 384]
            h = self.sat2_in_proj(tokens)                    # [B, 49, 64]
            attn_out, _ = self.sat2_mha(h, h, h)             # [B, 49, 64]
            h = self.sat2_ln(h + attn_out)                   # Residual + LN
            h_pool = h.mean(dim=1)                           # [B, 64]
            s_sat2 = self.sat2_proj(h_pool)                  # [B, sat2_dim]

        # Dynamic Gating: Core dynamically weights the specialist satellites
        alpha = torch.sigmoid(self.gate(s_core))
        fused = torch.cat([s_core, alpha[:, 0:1] * s_sat1, alpha[:, 1:2] * s_sat2], dim=-1)
        fused = self.fused_ln(fused)
        logits = self.classifier(fused)

        return logits, s_core, s_sat1, s_sat2, alpha
