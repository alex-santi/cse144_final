import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        # modify last layer of the model
        self.backbone = resnet18(weights=weights)
        
        #self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        

    def forward(self, x):
        return self.backbone(x)
