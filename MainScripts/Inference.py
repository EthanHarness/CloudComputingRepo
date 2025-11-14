import torch
from collections import OrderedDict

class Inference:
    def runValidations(self, modelType, model, validation_data, gpuLoc):
        model.eval()

        dev = torch.device(f"cuda:{gpuLoc}") if isinstance(gpuLoc, int) else torch.device(gpuLoc) if isinstance(gpuLoc, str) else gpuLoc

        with torch.no_grad():
            numberOfImages = 0
            correct_predictions = 0
            for batch_idx, (images, labels) in enumerate(validation_data):
                images = images.to(dev)
                labels = labels.to(dev)
                output = model(images)
                prediction = torch.argmax(output, dim=1)
                numberOfImages += labels.size(0)
                correct_predictions += (prediction == labels).sum().item()
        model.train()

        return torch.tensor([correct_predictions, numberOfImages])

            
def preparePyTorchDataloader(data, batch_size):
    from torch.utils.data import DataLoader
    return DataLoader(
        data,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False
    )
