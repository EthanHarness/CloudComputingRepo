import torch
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.models import resnet101
from torch.utils.data import Dataset, DataLoader

from MainScripts import Imagenet1kDataset
from MainScripts import Inference

print(f"GPUs {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
gpu_id = 0
print(f"Using device {torch.cuda.get_device_name(gpu_id)}")
torch.cuda.set_device(gpu_id)
location = f"cuda:{gpu_id}"

def createDataLoader(valid_set):
    return DataLoader(valid_set, batch_size=100, pin_memory=True, shuffle=False)

inf = Inference.Inference()
valid_set = Imagenet1kDataset.CustomImageNet1000("validation", False, -1)

print("Loading Models Snapshots......")
snapshotDeepSpeed = torch.load("./model/snapshot_DeepSpeed.pt", map_location=location)
snapshotPyTorch = torch.load("./model/snapshot_PyTorchDDP.pt", map_location=location)
print("Model Snapshots Loaded")

dsModel = resnet101(num_classes=1000)
ptModel = resnet101(num_classes=1000)

def clean_state_dict(state_dict):
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "module" in state_dict:
        state_dict = state_dict["module"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    return new_state_dict

dsModel.load_state_dict(clean_state_dict(snapshotDeepSpeed), strict=False)
ptModel.load_state_dict(clean_state_dict(snapshotPyTorch), strict=False)

dsModel.to(gpu_id)
dsResult = inf.runValidations("ds", dsModel, createDataLoader(valid_set), gpu_id)
dsModel.to("cpu")

ptModel.to(gpu_id)
ptResult = inf.runValidations("pt", ptModel, createDataLoader(valid_set), gpu_id)
ptModel.to("cpu")

print(f"Inference Results DeepSpeed: {dsResult[0].item()}/{dsResult[1].item()}={dsResult[0].item()/dsResult[1].item()}")
print(f"Inference Results PyTorch: {ptResult[0].item()}/{ptResult[1].item()}={ptResult[0].item()/ptResult[1].item()}")