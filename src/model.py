import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

# from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

class ViTB16Classifier(nn.Module):
    """ViT-B/16"""

    def __init__(self, num_classes=100, pretrained=True, dropout=0.3):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights)

        in_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.backbone.fc = self.backbone.heads
        self.backbone.layer4 = self.backbone.encoder.layers[-1]

    def forward(self, x):
        return self.backbone(x)
