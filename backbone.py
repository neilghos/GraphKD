import torch
import torch.nn as nn
import torchvision.models as models

class DualBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Backbone 1: ResNet-18 (CNN inductive bias)
        w1 = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.bb1 = models.resnet18(weights=w1)
        self.bb1.fc = nn.Identity()
        self.dim1 = 512

        # Backbone 2: Swin-T (Vision Transformer / Attention bias)
        w2 = models.Swin_T_Weights.DEFAULT if pretrained else None
        self.bb2 = models.swin_t(weights=w2)
        self.bb2.head = nn.Identity()
        self.dim2 = 768

    def forward(self, x):
        feat1 = self.bb1(x)  # [B, 512]
        feat2 = self.bb2(x)  # [B, 768]
        return feat1, feat2
