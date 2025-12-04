import torch
from torch.utils.data import DataLoader
import sys

from MainScripts import Imagenet1kDataset
from MainScripts import Inference
from MainScripts import ResnetModel
rm = ResnetModel.ActivationCheckpointingResnetModel()

print(f"GPUs {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
gpu_id = 0
print(f"Using device {torch.cuda.get_device_name(gpu_id)}")
torch.cuda.set_device(gpu_id)
location = f"cuda:{gpu_id}"

inf = Inference.Inference()
valid_set = Imagenet1kDataset.CustomImageNet1000("validation", False, 100, False)
valid_set = DataLoader(valid_set, batch_size=25, pin_memory=True, shuffle=False)

snapshotPyTorch = torch.load("./model/snapshot_PyTorchDDP.pt", map_location=location, weights_only=False)
ptModel = rm.createModel()
ptModel.load_state_dict(snapshotPyTorch["MODEL_STATE"])
print(f"PyTorch Epoch: {snapshotPyTorch['EPOCHS_RUN']}")
ptModel.to(gpu_id)
ptResult = inf.runValidations("pt", ptModel, valid_set, gpu_id)
ptModel.to("cpu")
print(f"Inference Results PyTorch: {ptResult[0].item()}/{ptResult[1].item()}={ptResult[0].item()/ptResult[1].item()*100}")
sys.stdout.flush()

snapshotDeepSpeed = torch.load("./model/snapshot_DeepSpeed.pt", map_location=location, weights_only=False)
dsModel = rm.createModel()
dsModel.load_state_dict(snapshotDeepSpeed)
dsModel.to(gpu_id)
ptResult = inf.runValidations("dp", dsModel, valid_set, gpu_id)
dsModel.to("cpu")
print(f"Inference Results DeepSpeed: {ptResult[0].item()}/{ptResult[1].item()}={ptResult[0].item()/ptResult[1].item()*100}")
sys.stdout.flush()
