#Mainly sourced from https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import psutil
from torchvision.models import resnet101
from torchvision.transforms import transforms
import sys

sys.path.append('/home/emh190004')
from CloudComputingRepo.MainScripts import CustomImageNet1000
from CloudComputingRepo.MainScripts import Inference
from CloudComputingRepo.MainScripts import PerformanceMonitor

TRAIN_SIZE = 100
VALIDATION_SIZE = 100
PERFORMANCE_FLAG = True
monitor = PerformanceMonitor("PyTorch")

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    pid = psutil.Process(os.getpid())
    print(f"Process {pid}")
    
def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

class PyTorchTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: Dataset,
        validation_data: Dataset,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        snapshot_path: str,
        batch_size: int
    ) -> None:
        self.inf = Inference()
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = prepare_dataloader(train_data, batch_size)
        self.validation_data = validation_data(validation_data, batch_size)
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[gpu_id])
        
        self.monitorCheck = PERFORMANCE_FLAG and self.gpu_id == 0
    
    def setMonitorStart(self):
        if self.monitorCheck: monitor.setPerfStartTime()
    
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        self.setMonitorStart()
        
        b_sz = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
        
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        if (self.monitorCheck): monitor.printEpochRuntime(epoch)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        if (self.monitorCheck): monitor.printTrainTimeStart()
        
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            
            self.setMonitorStart()
            self.inf.runValidations("pt", self.model, self.validation_data, self.gpu_id)
            if (self.monitorCheck): monitor.printValidationTime(epoch)
        
        if (self.monitorCheck): monitor.printTrainTimeEnd()
    

def load_train_objs():
    train_set = CustomImageNet1000("train", False, TRAIN_SIZE)
    valid_set = CustomImageNet1000("validation", False, VALIDATION_SIZE)
    model = resnet101(num_classes=train_set.getNumberOfClasses())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return train_set, valid_set, model, optimizer


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "../model/"):
    monitorCheck = PERFORMANCE_FLAG and rank==0
    setMonitorStart = lambda: monitor.setPerfStartTime() if (monitorCheck) else None
    
    if (monitorCheck): monitor.printStartTime()
     
    setMonitorStart()
    ddp_setup(rank, world_size)
    if (monitorCheck): monitor.printSetupTime()
    
    setMonitorStart()
    dataset, validDataset, model, optimizer = load_train_objs()
    if (monitorCheck): monitor.printLoadingTrainingTime()
    
    snapshot_path += "snapshot_PyTorchDDP.pt"
    
    setMonitorStart()
    trainer = PyTorchTrainer(model, train_data, validDataset, optimizer, rank, save_every, snapshot_path, batch_size)
    if (monitorCheck): monitor.printCreatingTrainingClass()
        
    setMonitorStart()
    trainer.train(total_epochs)
    if (monitorCheck): monitor.printTotalTrainingTime()
        
    if (monitorCheck): monitor.printEndTime()
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)