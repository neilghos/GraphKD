import torch
import torch.nn as nn
import timm

class DualBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Backbone 1: ConvNeXt-Base (Modern ConvNet inductive bias)
        self.bb1 = timm.create_model("convnext_base", pretrained=pretrained, num_classes=0)
        self.dim1 = 1024

        # Backbone 2: DINOv2 ViT-Base (Foundation Vision Transformer / Self-Attention bias)
        self.bb2 = timm.create_model("vit_base_patch14_dinov2", pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        self.dim2 = 768

    def forward(self, x):
        feat1 = self.bb1(x)  # [B, 1024]
        feat2 = self.bb2(x)  # [B, 768]
        return feat1, feat2
