#Mainly sourced from https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
#Modifications were made such as using gloo instead of nccl since we only have 1 gpu. 

import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import psutil
from datasets import load_dataset
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

def deepspeedSetup(rank: int):
    deepspeed.init_distributed()
    get_accelerator().set_device(rank)

    pid = psutil.Process(os.getpid())
    print(f"Process {pid}")

class DeepSpeedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)


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
        
        self.model.backward(loss)
        self.model.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

def load_train_objs():
    train_set = CustomImageNet1000("train", False, 100)
    model = resnet101(num_classes=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return train_set, model, optimizer


def main():
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--local_rank', type=int, help='Process local rank')
    parser.add_argument('--snapshot_path', default="../model/", type=str, help='Model snapshot save location')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    save_every = args.save_every
    total_epochs = args.total_epochs
    snapshot_path = args.snapshot_path + "snapshot_DeepSpeed.pt"

    deepspeedSetup(args.local_rank)

    dataset, model, optimizer = load_train_objs()
    ds_config = {
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "contiguous_gradients": True
        },
        "train_batch_size": args.batch_size * world_size,
        "train_micro_batch_size_per_gpu": args.batch_size,
    }

    modelEngine, optimizer, trainLoader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        optimizer=optimizer,
        config_params=ds_config
    )

    device = get_accelerator().device_name(args.local_rank)

    trainer = DeepSpeedTrainer(modelEngine, trainLoader, optimizer, args.local_rank, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    main()