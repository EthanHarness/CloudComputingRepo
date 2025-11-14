from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.distributed as dist
import torch
import os
from dotenv import load_dotenv

#This approach is more memory concious but has performance issues
#Spends a lot of time doing IO 
class CustomImageNet1000(Dataset):
    def __init__(self, dataType, force_recache=False, size=None):
        self.dataType = dataType
        if size == -1: size = None
        if dataType == "train" and size is None:
            self.length = 1281167
        elif dataType == "validation" and size is None:
            self.length = 50000
        elif dataType == "test" and size is None:
            self.length = 10000
        else:
            self.length = size
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

 
#Need to reduce the memory footprint of this for large datasets. 
#Initial IO is also very taxing
#Currently, every one loads all data into datalist. Fine for small datasizes, but for big datasets on multiple GPUs this gets very expensive
#Consider a caching approach where only a subset of the data is ever in datalist and we dynamically move data in and out for needs.     
"""
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

"""