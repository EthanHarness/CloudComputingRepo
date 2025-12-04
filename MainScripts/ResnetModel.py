import torch
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.models import resnet101

class CheckpointedResNet101(torch.nn.Module):
    def __init__(self, checkpoint_segments: int = 4, num_classes: int = 1000):
        super().__init__()
        base = resnet101(pretrained=False)
        base.fc = torch.nn.Linear(base.fc.in_features, num_classes)
        self.model = base
        self.checkpoint_segments = checkpoint_segments

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        layers = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        x = checkpoint_sequential(layers, self.checkpoint_segments, x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x


class ActivationCheckpointingResnetModel:
    def createModel(self):
        return CheckpointedResNet101()

    def testModelIsTheSame(self):
        savedModel = resnet101(num_classes=1000)
        snap = {"MODEL_STATE": savedModel.state_dict()}
        path = "./model/temp.pt"
        torch.save(snap, path)

        stateDict = torch.load(path, map_location="cpu")["MODEL_STATE"]
        m2 = resnet101(num_classes=1000)
        m2.load_state_dict(stateDict)
        m2.eval()
        
        newStateDict = {}
        for k, v in stateDict.items():
            newStateDict["model." + k] = v
        m1 = CheckpointedResNet101()
        m1.load_state_dict(newStateDict)
        m1.eval()

        incorrect = 0
        iterations = 100
        for num in range(iterations):
            x = torch.randn(2, 3, 224, 224, device="cpu", requires_grad=False)
            out1 = m1(x)
            out2 = m2(x)
            if not torch.equal(out1, out2):
                incorrect += 1
                print(f"Iteration {num} is incorrect")
        print(f"Correct percentage is {(iterations - incorrect)/iterations*100}%")