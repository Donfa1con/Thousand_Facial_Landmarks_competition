import torch.nn as nn
import torchvision.models as models

from .constants import NUM_PTS


class KeyPointNet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.pretrained = getattr(models, arch)(pretrained=True)
        if 'resnet' in arch:
            self.pretrained.fc = nn.Linear(self.pretrained.fc.in_features, 2 * NUM_PTS, bias=True)
        if 'mobilenet' in arch:
            self.pretrained.classifier[1] = nn.Linear(self.pretrained.classifier[1].in_features, 2 * NUM_PTS, bias=True)

    def forward(self, x):
        return self.pretrained(x)
