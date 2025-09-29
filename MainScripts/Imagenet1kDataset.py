from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.distributed as dist
import torch
import os
from dotenv import load_dotenv

class CustomImageNet1000(Dataset):
    def __init__(self, dataType, force_recache=False, size=None):
        self.dataType = dataType
        splitString = f"{dataType}" if size is None else f"{dataType}[:{size}]"
        self.dataList = load_dataset("ILSVRC/imagenet-1k", split=splitString, token=os.getenv("token"), cache_dir="~/scratch/hfCache")
        self.length = len(self.dataList) if size is None else size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.force_recache = force_recache

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        cache_path = os.path.expanduser(f"~/scratch/processedDataset/sample_{self.dataType}_{idx}.pt")
        if os.path.exists(cache_path) and not self.force_recache:
            cache = torch.load(cache_path)
            return cache["image"], cache["label"]

        item = self.dataList[idx]
        image = item["image"]
        label = item["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")
            
        image = self.transform(image)
        torch.save({"image": image, "label": label}, cache_path)
        return image,label
    
    def getNumberOfClasses(self):
        return 1000

    