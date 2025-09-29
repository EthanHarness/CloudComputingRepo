from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch
import os
import sys

sys.path.append('/home/emh190004')
from CloudComputingRepo.MainScripts import Imagenet1kDataset as ds



def forceRecache():
    def collate_fn(batch):
        return {
            "images": torch.stack([x[0] for x in batch]),
            "labels": torch.tensor([x[1] for x in batch])
        }

    trainSet = ds.CustomImageNet1000("train", True)
    trainDataLoader = DataLoader(trainSet, batch_size=1024, num_workers=16, collate_fn=collate_fn)
    for batch_data in trainDataLoader:
        print(batch_data)

    validationSet = ds.CustomImageNet1000("validation", True)
    validationSetDataLoader = DataLoader(validationSet, batch_size=1024, num_workers=16, collate_fn=collate_fn)
    for batch_data in validationSetDataLoader:
        print(batch_data)

    testSet = ds.CustomImageNet1000("test", True)
    testDataLoader = DataLoader(testSet, batch_size=1024, num_workers=16, collate_fn=collate_fn)
    for batch_data in testDataLoader:
        print(batch_data)

def main():
    forceRecache()

if __name__ == "__main__":
    main()