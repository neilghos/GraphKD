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
            if not pretrained or attempt == retries - 1:
                raise e
            print(f"[timm] Download/DNS blip for '{model_name}' ({e}). Resetting HTTP session & retrying ({attempt+1}/{retries})...")
            try:
                from huggingface_hub.utils._http import close_session
                close_session()
            except Exception:
                pass
            time.sleep(2)

class DualReconstructionStudent(nn.Module):
    """
    Dual-Reconstruction Student for Multi-Teacher Knowledge Distillation (Option B).
    
    Architecture (Unified 128x128 Pretrained Resolution):
      - Shared Core Trunk: ConvNeXt-Femto (384d, ~4.83M params, 128x128) -> core_norm (384d)
      - ConvNeXt Specialist Island: MobileNetV3-Small-075 (1024d -> 640d, ~1.02M params, 128x128) -> spec_conv (640d)
        Full-Basis Reconstruction: recon_conv([spec_conv (640d), core_norm (384d)]) -> emb_conv (1024d)
        Head: Linear(1024, 200) (Teacher-Probe Initialized) -> logits_conv
      - DINOv2 Specialist Island: MobileViT-XXS (320d -> 384d, ~0.95M params, 128x128) -> spec_dino (384d)
        Full-Basis Reconstruction: recon_dino([spec_dino (384d), core_norm (384d)]) -> emb_dino (768d)
        Head: Linear(768, 200)  (Teacher-Probe Initialized) -> logits_dino
    """
    def __init__(self, num_classes=200, pretrained=True):
        super().__init__()
        # 1. Instantiate Pretrained Core + CNN Island + Attention Island (intact stems)
        self.trunk = create_timm_model_safe("convnext_femto", pretrained=pretrained, num_classes=0)
        self.conv_island = create_timm_model_safe("mobilenetv3_small_075", pretrained=pretrained, num_classes=0)
        self.dino_island = create_timm_model_safe("mobilevit_xxs", pretrained=pretrained, num_classes=0)

        # Infer exact output feature dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            trunk_dim = self.trunk(dummy).shape[-1]              # 384
            conv_island_dim = self.conv_island(dummy).shape[-1]  # 1024
            dino_island_dim = self.dino_island(dummy).shape[-1]  # 320

        self.conv_spec_dim = 1024 - trunk_dim  # 640 dims
        self.dino_spec_dim = 768 - trunk_dim   # 384 dims

        # Shared Core Normalization
        self.core_ln = nn.LayerNorm(trunk_dim)

        # 2. ConvNeXt Specialist Extraction (640d) + Full-Basis 1024d Reconstruction
        self.conv_proj = nn.Sequential(
            nn.LayerNorm(conv_island_dim),
            nn.Linear(conv_island_dim, self.conv_spec_dim),
            nn.GELU()
        )
        self.recon_conv = nn.Linear(1024, 1024)
        self.head_conv = nn.Linear(1024, num_classes)

        # 3. DINOv2 Specialist Extraction (384d) + Full-Basis 768d Reconstruction
        self.dino_proj = nn.Sequential(
            nn.LayerNorm(dino_island_dim),
            nn.Linear(dino_island_dim, self.dino_spec_dim),
            nn.GELU()
        )
        self.recon_dino = nn.Linear(768, 768)
        self.head_dino = nn.Linear(768, num_classes)

    def forward(self, x):
        # Single unified 128x128 resolution for all 3 backbones (intact stems, 3x faster than 224x224)
        if x.shape[-1] != 128:
            x_128 = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        else:
            x_128 = x

        # 1. Forward through Core + CNN Island + Attention Island
        shared_core = self.core_ln(self.trunk(x_128))          # [B, 384]
        raw_conv = self.conv_island(x_128)                     # [B, 1024]
        raw_dino = self.dino_island(x_128)                     # [B, 320]

        # 2. ConvNeXt Stream Reconstruction ([640d Specialist, 384d Core] -> Full 1024d Teacher Basis)
        spec_conv = self.conv_proj(raw_conv)                   # [B, 640]
        emb_conv = self.recon_conv(torch.cat([spec_conv, shared_core], dim=-1))  # [B, 1024]
        logits_conv = self.head_conv(emb_conv)                 # [B, 200]

        # 3. DINOv2 Stream Reconstruction ([384d Specialist, 384d Core] -> Full 768d Teacher Basis)
        spec_dino = self.dino_proj(raw_dino)                   # [B, 384]
        emb_dino = self.recon_dino(torch.cat([spec_dino, shared_core], dim=-1))  # [B, 768]
        logits_dino = self.head_dino(emb_dino)                 # [B, 200]

        # 4. Combined Ensemble Logits & Joint 1792d Representation
        logits_ens = 0.5 * logits_conv + 0.5 * logits_dino
        emb_joint = torch.cat([emb_conv, emb_dino], dim=-1)    # [B, 1792]

        return logits_conv, logits_dino, logits_ens, emb_conv, emb_dino, emb_joint
