#https://stackoverflow.com/questions/78559358/i-want-to-train-a-resnet-50-model-for-an-image-clasification-task-how-do-i-modi

import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()

        # Load the pretrained ResNet50 model
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # don't use layer4 and the final fully connected layer
        self.backbone = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.avgpool,
        )

        # save memory
        del self.resnet

        # Define the new fully connected layers, maybe with a bit more compute
        # since we removed a few resnet layers
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes),
            nn.Dropout(0.2),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


def getCustomResNetModel():
    #Create Custom ResNet50 model
    num_classes = 2 #Hard Coded based on Dataset
    dim = 240 #Hard Coded based on Dataset
    model = CustomResNet50(num_classes=num_classes)
    x = torch.randn(2, 3, dim, dim)
    _ = model(x)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return [model, loss_fn, optimizer]
    