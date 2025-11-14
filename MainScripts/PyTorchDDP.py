import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import os
import psutil

from Imagenet1kDataset import CustomImageNet1000
from Inference import Inference
from PerformanceMonitor import PerformanceMonitor
from ResnetModel import ActivationCheckpointingResnetModel

TRAIN_SIZE = 131072
VALIDATION_SIZE = -1
PERFORMANCE_FLAG = True
MEMORY_PROFILING_FLAG = False
ENABLE_SAVING = True
RUN_VALIDATIONS = True
monitor = PerformanceMonitor("PyTorch")

def ddp_setup(rank, world_size):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    pid = psutil.Process(os.getpid())
    print(f"Process {pid}")
    
def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
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
        batch_size: int,
        profiler
    ) -> None:
        self.inf = Inference()
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = prepare_dataloader(train_data, batch_size)
        self.validation_data = prepare_dataloader(validation_data, batch_size)
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.profiler = profiler
        if os.path.exists(snapshot_path) and ENABLE_SAVING:
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[gpu_id])
        
        self.monitorCheck = PERFORMANCE_FLAG and self.gpu_id == 0
        self.profilingCheck = MEMORY_PROFILING_FLAG and self.gpu_id == 0
    
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
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for batchIndex, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            
            self._run_batch(source, targets)
        
        if (self.monitorCheck): 
            monitor.printEpochRuntime(epoch)
            monitor.flushOutput()

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
            if ENABLE_SAVING and self.gpu_id == 0 and epoch % self.save_every == 0: self._save_snapshot(epoch)
            if (epoch % self.save_every == 0 and RUN_VALIDATIONS): self.runValidations(epoch)
                
            if self.profilingCheck and epoch < monitor.getProfilerSteps(): self.profiler.step()
            if self.profilingCheck and epoch == monitor.getProfilerSteps() - 1: self.profiler.stop()
        
        if ENABLE_SAVING and self.gpu_id == 0 and epoch % self.save_every != 0: self._save_snapshot(max_epochs-1)
        if (self.monitorCheck): 
            monitor.printTrainTimeEnd()
            monitor.flushOutput()
            
    def runValidations(self, epoch):
        self.setMonitorStart()
        outputTensor = self.inf.runValidations("pt", self.model, self.validation_data, self.gpu_id)
        if (self.monitorCheck): monitor.printValidationTime(epoch)
        outputTensor = outputTensor.to(f'cuda:{self.gpu_id}')
        
        if self.gpu_id == 0:
            gathered_data = [torch.zeros_like(outputTensor) for _ in range(dist.get_world_size())]
            dist.gather(outputTensor, gather_list=gathered_data, dst=0)
            gathered_data = torch.sum(torch.stack(gathered_data, dim=0), dim=0)
            print(f"Inference Results: {gathered_data[0].item()}/{gathered_data[1].item()}={gathered_data[0].item()/gathered_data[1].item()*100}%")
        else:
            dist.gather(outputTensor, dst=0)
            
    

def load_train_objs():
    train_set = CustomImageNet1000("train", False, TRAIN_SIZE)
    valid_set = CustomImageNet1000("validation", False, VALIDATION_SIZE)
    model = ActivationCheckpointingResnetModel().createModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    return train_set, valid_set, model, optimizer


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "../model/"):
    monitorCheck = PERFORMANCE_FLAG and rank==0
    setMonitorStart = lambda: monitor.setPerfStartTime() if (monitorCheck) else None
    
    if (monitorCheck): monitor.printStartTime()
    ddp_setup(rank, world_size)
    dataset, validDataset, model, optimizer = load_train_objs()
    
    snapshot_path += "snapshot_PyTorchDDP.pt"
    profiler = None
    if (MEMORY_PROFILING_FLAG): profiler = monitor.createProfiler(rank)
    trainer = PyTorchTrainer(model, dataset, validDataset, optimizer, rank, save_every, snapshot_path, batch_size, profiler)
        
    setMonitorStart()
    trainer.train(total_epochs)
    if (monitorCheck): 
        monitor.printTotalTrainingTime()
        monitor.printEndTime()

    if (MEMORY_PROFILING_FLAG and rank == 0):
        torch.cuda.synchronize()
        monitor.exportMemory(profiler)
    destroy_process_group()


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    args = parser.parse_args()
    
    print(f"WORLD_SIZE: {torch.cuda.device_count()} RANK {rank}")
    if rank == 0:
        print(f"Number of available GPUs: {world_size}")
        for i in range(world_size):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    main(rank, world_size, args.save_every, args.total_epochs, args.batch_size)