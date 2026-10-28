import torch
import torch.nn as nn
import torchvision.models as models

class ModularStudent(nn.Module):
    def __init__(self, core_dim=128, sat1_dim=32, sat2_dim=16, num_classes=200):
        super().__init__()
        # 1. Medium Core Trunk (MobileNetV3-Small: ~1.5M params)
        mb = models.mobilenet_v3_small(weights=None)
        self.trunk = mb.features
        trunk_dim = 576

        # Homophilic Core Head (128-dim invariant semantic core)
        self.core_head = nn.Linear(trunk_dim, core_dim)

        # 2. Small Parallel Satellites for Heterophilic Specialist Regions
        self.sat1_head = nn.Sequential(nn.Linear(trunk_dim, 64), nn.ReLU(), nn.Linear(64, sat1_dim))
        self.sat2_head = nn.Sequential(nn.Linear(trunk_dim, 64), nn.ReLU(), nn.Linear(64, sat2_dim))

        # 3. Dynamic Gating & Readout
        self.gate = nn.Linear(core_dim, 2)
        self.classifier = nn.Linear(core_dim + sat1_dim + sat2_dim, num_classes)

    def forward(self, x):
        feat = self.trunk(x).mean(dim=[-2, -1])  # [B, 576]
        
        s_core = self.core_head(feat)
        s_sat1 = self.sat1_head(feat)
        s_sat2 = self.sat2_head(feat)

        # Dynamic Gating: Core dynamically weights the specialist satellites
        alpha = torch.sigmoid(self.gate(s_core))
        fused = torch.cat([s_core, alpha[:, 0:1] * s_sat1, alpha[:, 1:2] * s_sat2], dim=-1)
        logits = self.classifier(fused)

        return logits, s_core, s_sat1, s_sat2, alpha
