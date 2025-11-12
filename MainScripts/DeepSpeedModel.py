import torch
from torchvision.models import resnet101
import deepspeed
from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint

class CheckpointedResNet101(torch.nn.Module):
    def __init__(self, checkpoint_segments: int = 4, num_classes: int = 1000):
        super().__init__()
        self.model = resnet101(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.checkpoint_segments = checkpoint_segments

    def _all_blocks(self):
        return list(self.model.layer1.children()) + \
               list(self.model.layer2.children()) + \
               list(self.model.layer3.children()) + \
               list(self.model.layer4.children())

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        blocks = self._all_blocks()
        num_blocks = len(blocks)
        if self.checkpoint_segments > 0:
            segment_size = num_blocks // self.checkpoint_segments
        else:
            segment_size = num_blocks

        for i in range(0, num_blocks, segment_size):
            segment = blocks[i:i + segment_size]

            def run_segment(*inputs, segment=segment):
                out = inputs[0]
                for block in segment:
                    out = block(out)
                return out

            x = checkpoint(run_segment, x)


        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x

class CreateCustomDeepSpeedResnetModel:
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