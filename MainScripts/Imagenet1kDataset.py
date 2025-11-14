from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch
import os
from numpy.random import default_rng
from typing import List, Tuple

TRAIN_SPACE_MAX = 1281167
VALID_SPACE_MAX = 50000
TEST_SPACE_MAX = 100000

class CustomImageNet1000(Dataset):
    def __init__(self, dataType, force_recache=False, size=None, shuffle=True):
        self.dataType = dataType
        if dataType == "train" and (size is None or size == -1):
            self.length = TRAIN_SPACE_MAX
        elif dataType == "validation" and (size is None or size == -1):
            self.length = VALID_SPACE_MAX
        elif dataType == "test" and (size is None or size == -1):
            self.length = TEST_SPACE_MAX
        else:
            self.length = size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.dataList = self.createDataList(shuffle, force_recache)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.dataList[idx][0],self.dataList[idx][1]

    def createDataList(self, shuffle: bool, force_recache: bool) -> List[Tuple]:
        if self.dataType == "train": space = TRAIN_SPACE_MAX
        elif self.dataType == "validation": space = VALID_SPACE_MAX
        else: space = TEST_SPACE_MAX

        if shuffle:
            rng = default_rng()
            numbers = rng.choice(space, size=self.length, replace=False).tolist()
        else: numbers = [x for x in range(self.length)]
        
        data = []
        flag = False
        for x in numbers:
            cache_path = os.path.expanduser(f"~/scratch/processedDataset/sample_{self.dataType}_{x}.pt")
            if not os.path.exists(cache_path) or force_recache:
                flag = True
                break

            cache = torch.load(cache_path)
            data.append(cache)
        if not flag: return [(x["image"],x["label"]) for x in data]

        print(f"REBUILDING CACHE FOR {self.dataType}")
        maxNum = max(numbers)
        splitString = f"{self.dataType}[:{maxNum}]"
        data = load_dataset("ILSVRC/imagenet-1k", split=splitString, token=os.getenv("token"))
        for item in data:
            if item["image"].mode != "RGB": item["image"] = item["image"].convert("RGB")
            item["image"] = self.transform(item["image"])
            torch.save({"image": item["image"], "label": item["label"]})

        #Data has every entry from 0 up to our max value. 
        #If shuffle is true we still need to rearrange data
        if not shuffle: return [(x["image"],x["label"]) for x in data]
        return [(data[x]["image"],data[x]["label"]) for x in numbers]

    def getNumberOfClasses(self):
        return 1000

    