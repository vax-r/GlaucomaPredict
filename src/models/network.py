import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class GlaucomaNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)