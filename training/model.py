import torch.nn as nn
from torchvision import models


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10, dropout=0.2, pretrained=False, freeze_backbone=False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)


def build_model(config):
    if config.get("model_name", "resnet18") != "resnet18":
        raise ValueError("Only resnet18 is implemented in this project template.")
    return ResNet18CIFAR(
        num_classes=int(config.get("num_classes", 10)),
        dropout=float(config.get("dropout", 0.2)),
        pretrained=bool(config.get("pretrained", False)),
        freeze_backbone=bool(config.get("freeze_backbone", False)),
    )
