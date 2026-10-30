import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def create_timm_model_safe(model_name, pretrained=True, num_classes=0, retries=5):
    """Safely creates a timm model, resetting the HuggingFace httpx session if a DNS blip occurs."""
    for attempt in range(retries):
        try:
            return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        except Exception as e:
            if not pretrained or "No pretrained weights exist" in str(e) or attempt == retries - 1:
                raise e
            print(f"[timm] Download/DNS blip for '{model_name}' ({e}). Resetting HTTP session & retrying ({attempt+1}/{retries})...")
            try:
                from huggingface_hub.utils._http import close_session
                close_session()
            except Exception:
                pass
            time.sleep(2)

class LowRankBasisMixer(nn.Module):
    """
    Parameter-efficient full-basis reconstruction layer (Residual + Low-Rank Cross-Channel Mixing).
    Allows every single teacher channel (all 1024 ConvNeXt or 768 DINOv2 channels) to be reconstructed
    from BOTH the Private Specialist Island and the Shared Core using ~4x fewer parameters than a dense matrix.
    """
    def __init__(self, dim, rank=128):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, dim))
        self.bias = nn.Parameter(torch.zeros(1, dim))
        self.down = nn.Linear(dim, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, dim, bias=False)

    def forward(self, x):
        return self.scale * x + self.up(self.act(self.down(x))) + self.bias

class DualReconstructionStudent(nn.Module):
    """
    Ultra-Compact Dual-Reconstruction Student (~7.47M Total Params, ~23.4x Compression).
    
    Architecture (Unified 128x128 Pretrained Resolution, 85% Island / 15% Core Split):
      - Shared Hybrid Core Trunk (15.2% of backbone params):
          mobilevit_xxs (CNN+Attention Hybrid, ~0.95M params, 320d) -> shared_core (320d)
      - ConvNeXt Specialist Island (54.0% of backbone params):
          convnext_atto (Pure ConvNeXt Specialist, ~3.38M params, 75.7% IN-1K, 320d -> 704d) -> spec_conv (704d)
          Reconstructed ConvNeXt = recon_conv([spec_conv (704d), shared_core (320d)]) -> emb_conv (1024d)
          Head: Linear(1024, 200) (Teacher-Probe Initialized) -> logits_conv
      - DINOv2 Specialist Island (30.8% of backbone params):
          mobilevit_xs (Scaled ViT/Attention Specialist, ~1.93M params, 74.8% IN-1K, 384d -> 448d) -> spec_dino (448d)
          Reconstructed DINOv2   = recon_dino([spec_dino (448d), shared_core (320d)]) -> emb_dino (768d)
          Head: Linear(768, 200)  (Teacher-Probe Initialized) -> logits_dino
    """
    def __init__(self, num_classes=200, pretrained=True):
        super().__init__()
        # 1. Compact Hybrid Core (0.95M) + Pure ConvNeXt-Atto Island (3.38M) + Scaled MobileViT-XS Island (1.93M)
        self.trunk = create_timm_model_safe("mobilevit_xxs", pretrained=pretrained, num_classes=0)
        self.conv_island = create_timm_model_safe("convnext_atto", pretrained=pretrained, num_classes=0)
        self.dino_island = create_timm_model_safe("mobilevit_xs", pretrained=pretrained, num_classes=0)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            trunk_dim = self.trunk(dummy).shape[-1]              # 320
            conv_island_dim = self.conv_island(dummy).shape[-1]  # 320
            dino_island_dim = self.dino_island(dummy).shape[-1]  # 384

        self.conv_spec_dim = 1024 - trunk_dim  # 704 dims (69% Specialist / 31% Core)
        self.dino_spec_dim = 768 - trunk_dim   # 448 dims (58% Specialist / 42% Core)

        # Shared Core Normalization (320d)
        self.core_ln = nn.LayerNorm(trunk_dim)

        # 2. ConvNeXt Specialist Extraction (320d -> 704d) + Full-Basis 1024d Reconstruction
        self.conv_proj = nn.Sequential(
            nn.LayerNorm(conv_island_dim),
            nn.Linear(conv_island_dim, self.conv_spec_dim),
            nn.GELU()
        )
        self.recon_conv = LowRankBasisMixer(1024, rank=128)
        self.head_conv = nn.Linear(1024, num_classes)

        # 3. DINOv2 Specialist Extraction (384d -> 448d) + Full-Basis 768d Reconstruction
        self.dino_proj = nn.Sequential(
            nn.LayerNorm(dino_island_dim),
            nn.Linear(dino_island_dim, self.dino_spec_dim),
            nn.GELU()
        )
        self.recon_dino = LowRankBasisMixer(768, rank=128)
        self.head_dino = nn.Linear(768, num_classes)

    def forward(self, x):
        if x.shape[-1] != 128:
            x_128 = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        else:
            x_128 = x

        # 1. Forward through Compact Hybrid Core + ConvNeXt-Atto Island + MobileViT-XS Island
        shared_core = self.core_ln(self.trunk(x_128))          # [B, 320]
        raw_conv = self.conv_island(x_128)                     # [B, 320]
        raw_dino = self.dino_island(x_128)                     # [B, 384]

        # 2. ConvNeXt Stream Reconstruction ([704d Specialist, 320d Core] -> Full 1024d Teacher Basis)
        spec_conv = self.conv_proj(raw_conv)                   # [B, 704]
        emb_conv = self.recon_conv(torch.cat([spec_conv, shared_core], dim=-1))  # [B, 1024]
        logits_conv = self.head_conv(emb_conv)                 # [B, 200]

        # 3. DINOv2 Stream Reconstruction ([448d Specialist, 320d Core] -> Full 768d Teacher Basis)
        spec_dino = self.dino_proj(raw_dino)                   # [B, 448]
        emb_dino = self.recon_dino(torch.cat([spec_dino, shared_core], dim=-1))  # [B, 768]
        logits_dino = self.head_dino(emb_dino)                 # [B, 200]

        # 4. Combined Ensemble Logits & Joint 1792d Representation
        logits_ens = 0.5 * logits_conv + 0.5 * logits_dino
        emb_joint = torch.cat([emb_conv, emb_dino], dim=-1)    # [B, 1792]

        return logits_conv, logits_dino, logits_ens, emb_conv, emb_dino, emb_joint
