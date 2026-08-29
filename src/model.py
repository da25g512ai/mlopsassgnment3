import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    if architecture.lower() == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")