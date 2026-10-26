"""
Convolutional Neural Network architecture for 28x28 PneumoniaMNIST chest X-rays.
Designed for high efficiency, clear feature representations, and reliable Grad-CAM explainability.
"""
import torch
import torch.nn as nn
from typing import Dict, Any


class ChestXRayMiniCNN(nn.Module):
    """
    Compact 3-stage CNN tailored for 28x28 single-channel Chest X-ray images.
    Features batch normalization and dropout for regularized learning.
    Target layer for Grad-CAM is accessible directly as `conv3`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stage 1: (B, 1, 28, 28) -> (B, 32, 14, 14)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2: (B, 32, 14, 14) -> (B, 64, 7, 7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3: (B, 64, 7, 7) -> (B, 128, 3, 3) (Primary Grad-CAM target layer)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier head
        self.global_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw classification logits.
        Input: (B, in_channels, 28, 28)
        Output: (B, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """Returns the final convolutional layer suitable for Grad-CAM feature attribution."""
        return self.conv3


def build_model(config: Dict[str, Any]) -> ChestXRayMiniCNN:
    """Builds and initializes the model according to configuration."""
    model_cfg = config.get("model", {})
    in_channels = model_cfg.get("in_channels", 1)
    num_classes = model_cfg.get("num_classes", 2)
    dropout = model_cfg.get("dropout", 0.25)
    return ChestXRayMiniCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
    )
