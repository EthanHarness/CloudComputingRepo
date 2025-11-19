from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.distributed as dist
import torch
import os
import numpy as np

TRAIN_SPACE_MAX = 1281167
VALID_SPACE_MAX = 50000
TEST_SPACE_MAX = 100000

class CustomImageNet1000(Dataset):
    def __init__(self, dataType, force_recache=False, size=None):
        self.dataType = dataType
        if dataType == "train" and (size is None or size == -1):
            self.length = TRAIN_SPACE_MAX
            self.space = TRAIN_SPACE_MAX
        elif dataType == "validation" and (size is None or size == -1):
            self.length = VALID_SPACE_MAX
            self.space = VALID_SPACE_MAX
        elif dataType == "test" and (size is None or size == -1):
            self.length = TEST_SPACE_MAX
            self.space = TEST_SPACE_MAX
        else:
            self.space = TRAIN_SPACE_MAX if (dataType == "train") else VALID_SPACE_MAX if (dataType == "validation") else TEST_SPACE_MAX
            self.length = size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.force_recache = force_recache
        rng = np.random.default_rng(seed=0)
        self.numbers = rng.choice(self.space, size=self.length, replace=False).tolist()

    def __len__(self):
        return self.length
    
    def setSeed(self, val: int):
        rng = np.random.default_rng(seed=val)
        self.numbers = rng.choice(self.space, size=self.length, replace=False).tolist()

    def __getitem__(self, numbersOffset):
        idx = self.numbers[numbersOffset]
        cache_path = os.path.expanduser(f"~/scratch/processedDataset/sample_{self.dataType}_{idx}.pt")
        if os.path.exists(cache_path) and not self.force_recache:
            cache = torch.load(cache_path)
            return cache["image"], cache["label"]
        
        splitString = f"{self.dataType}" if self.length is None else f"{self.dataType}[:{self.length}]"
        self.dataList = load_dataset("ILSVRC/imagenet-1k", split=splitString, token=os.getenv("token"))
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