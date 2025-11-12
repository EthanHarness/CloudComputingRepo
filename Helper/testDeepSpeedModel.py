import torch
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.models import resnet101
import sys

from MainScripts import DeepSpeedModel
ckClass = DeepSpeedModel.CreateCustomResnetModel()
ckClass.testModelIsTheSame()
