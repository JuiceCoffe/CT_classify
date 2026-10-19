"""基础 TinyCNN，与任务书的三层二维网络一致。"""
import torch
from torch import nn


class TinyCNN(nn.Module):
    """输入 float32 [B,1,28,28]，输出 [B] 原始 logits（不是概率）。"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, 1)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1:] != (1, 28, 28) or image.dtype != torch.float32:
            raise ValueError(f'TinyCNN image expected float32 [B,1,28,28], got {image.dtype} {image.shape}')
        return self.classifier(self.pool(self.features(image)).flatten(1)).squeeze(1)
