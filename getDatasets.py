import os
import math
import random
import torch
import pandas
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, train, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB') # Ensure 3 channels for JPGs
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        # You can also return a label if available, e.g., from a separate list
        # For simplicity, we are only returning the image here.
        return image, label


def get_files_in_directory(directory_path):
    files = []
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path) and full_path.find(":Zone.Identifier") == -1:
            files.append(full_path)
    return files

def get_training_test_data(directory):
    files = get_files_in_directory(directory)
    files = random.sample(files, len(files))
    trainingLen = math.ceil(len(files) * 0.80)
    trainingData = files[:trainingLen]
    testData = files[trainingLen:]
    return [trainingData, testData]

def get_training_test_labels(training, test, labelCSV):
    trainLabel = []
    testLabel = []
    csvFile = pandas.read_csv(labelCSV)
    labelDict = csvFile.set_index("Image").apply(lambda row: row.tolist(), axis=1).to_dict()

    for imgLabel in training:
        idx1 = imgLabel.find("Image")
        idx2 = imgLabel.find(".jpg")
        trainLabel.append(labelDict[imgLabel[idx1:idx2]][0])

    for imgLabel in test:
        idx1 = imgLabel.find("Image")
        idx2 = imgLabel.find(".jpg")
        testLabel.append(labelDict[imgLabel[idx1:idx2]][0])
    
    return [trainLabel, testLabel]


def getDatas():
    imgDirectory = "./data/BrainTumor/BrainTumor"
    labelCSV = "./data/BrainTumor/Brain Tumor.csv"
    [training, test] = get_training_test_data(imgDirectory)
    [trainLabels, testLabels] = get_training_test_labels(training, test, labelCSV)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CustomImageDataset(training, trainLabels, True, transform)
    test_dataset = CustomImageDataset(test, testLabels, False, transform)
    return [train_dataset, test_dataset]