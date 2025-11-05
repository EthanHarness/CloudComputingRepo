import torch
from collections import OrderedDict
import sys
sys.path.append('/home/emh190004')
from CloudComputingRepo.MainScripts import Imagenet1kDataset as ds

class Inference:
    def runValidations(self, modelType, model, sz, gpuLoc, batch_size):
        if sz == -1: sz = None
        validation_set = ds.CustomImageNet1000("validation", sz)

        if modelType == "deepspeed": path = "../model/snapshot_DeepSpeed.pt"
        elif modelType == "pytorch": path = "../model/snapshot_PyTorchDDP.pt"
        else: path = ""
        
        if modelType == "pytorch" or modelType == "pt":
            validation_set = preparePyTorchDataloader(validation_set, batch_size)

        if path != "":
            print("Inferencing from file")

            if isinstance(gpuLoc, int): map_loc = torch.device(f"cuda:{gpuLoc}")
            elif isinstance(gpuLoc, str): map_loc = torch.device(gpuLoc)
            else: map_loc = gpuLoc

            ck = torch.load(path, map_location=map_loc)
            state_dict = ck["MODEL_STATE"]
            sd_keys = list(state_dict.keys())
            model_keys = list(model.state_dict().keys())

            def key_has_module(keys):
                return len(keys) > 0 and keys[0].startswith("module.")

            sd_has_module = key_has_module(sd_keys)
            model_has_module = key_has_module(model_keys)

            if sd_has_module and not model_has_module:
                new_state_dict = OrderedDict((k.replace("module.", "", 1), v) for k, v in state_dict.items())
            elif not sd_has_module and model_has_module:
                new_state_dict = OrderedDict((f"module.{k}", v) for k, v in state_dict.items())
            else:
                new_state_dict = state_dict

            model.load_state_dict(new_state_dict, strict=False)
        
        else: print("Inferencing from model directly")

        model.eval()

        dev = torch.device(f"cuda:{gpuLoc}") if isinstance(gpuLoc, int) else torch.device(gpuLoc) if isinstance(gpuLoc, str) else gpuLoc

        with torch.no_grad():
            for source, targets in validation_set:
                source = source.to(dev)
                output = model(source)
                print(output, targets)

        if path == "":
            model.train()
            
def preparePyTorchDataloader(data, batch_size):
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import DataLoader
    return DataLoader(
        data,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )
