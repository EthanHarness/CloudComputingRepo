import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as log
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
import psutil
from torchvision.models import resnet101
from torch.utils.checkpoint import checkpoint_sequential
import sys

from Imagenet1kDataset import CustomImageNet1000
from Inference import Inference
from PerformanceMonitor import PerformanceMonitor
from DeepSpeedModel import CreateCustomDeepSpeedResnetModel

TRAIN_SIZE = 1000
VALIDATION_SIZE = 100
PERFORMANCE_FLAG = True
MEMORY_PROFILING_FLAG = False
ENABLE_SAVING = False
monitor = PerformanceMonitor("DeepSpeed")

def deepspeedSetup(rank: int):
    deepspeed.init_distributed()
    get_accelerator().set_device(rank)

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
    
class DeepSpeedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
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
        self.train_data = train_data
        self.validation_data = prepare_dataloader(validation_data, batch_size)
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.profiler = profiler
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
            
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
        model = self.model
        output = model(source)
        self.optimizer.zero_grad()
        loss = F.cross_entropy(output, targets)
        self.model.backward(loss)
        self.model.step()

    def _run_epoch(self, epoch):
        self.setMonitorStart()
        
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for batchIndex, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
        
        if (self.monitorCheck): 
            monitor.printEpochRuntime(epoch)
            monitor.flushOutput()
            

    def _save_snapshot(self, epoch):
        if not ENABLE_SAVING: return
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
                
            if self.profilingCheck and epoch < monitor.getProfilerSteps(): self.profiler.step()
            if self.profilingCheck and epoch == monitor.getProfilerSteps() - 1:
                self.profiler.stop()
                torch.cuda.synchronize()
                monitor.exportMemory(self.profiler)
                self.profiler = None
            
        if (self.monitorCheck): 
            monitor.printTrainTimeEnd()
            monitor.flushOutput()

    def runValidations(self):
        self.setMonitorStart()
        outputTensor = self.inf.runValidations("dp", self.model, self.validation_data, self.gpu_id)
        if (self.monitorCheck): monitor.printValidationTime(epoch)
        outputTensor = outputTensor.to(f'cuda:{self.gpu_id}')
        
        if self.gpu_id == 0:
            gathered_data = [torch.zeros_like(outputTensor) for _ in range(dist.get_world_size())]
            dist.gather(outputTensor, gather_list=gathered_data, dst=0)
            gathered_data = torch.sum(torch.stack(gathered_data, dim=0), dim=0)
            print(f"Inference Results: {gathered_data[0].item()}/{gathered_data[1].item()}={gathered_data[0].item()/gathered_data[1].item()}")
        else:
            dist.gather(outputTensor, dst=0)
        
def load_train_objs():
    train_set = CustomImageNet1000("train", False, TRAIN_SIZE)
    valid_set = CustomImageNet1000("validation", False, VALIDATION_SIZE)
    model = CreateCustomDeepSpeedResnetModel().createModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return train_set, valid_set, model, optimizer


def main():
    import warnings
    warnings.filterwarnings("ignore", message=".*weights_only=False.*")
    
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--local_rank', type=int, help='Process local rank')
    parser.add_argument('--snapshot_path', default="./model/", type=str, help='Model snapshot save location')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    save_every = args.save_every
    total_epochs = args.total_epochs
    snapshot_path = args.snapshot_path + "snapshot_DeepSpeed.pt"

    monitorCheck = PERFORMANCE_FLAG and args.local_rank == 0
    setMonitorStart = lambda: monitor.setPerfStartTime() if (monitorCheck) else None
    
    profiler = None
    if (MEMORY_PROFILING_FLAG): profiler = monitor.createProfiler(args.local_rank)
    if (monitorCheck): monitor.printStartTime()
    
    deepspeedSetup(args.local_rank)
    dataset, validation_data, model, optimizer = load_train_objs()
        
    ds_config = {
        "train_batch_size": args.batch_size * world_size,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "none"},
            "offload_optimizer": {"device": "none"},
            "allgather_partitions": False,
            "reduce_scatter": True,
            "contiguous_gradients": False,
            "overlap_comm": True
        },
        "gradient_accumulation_steps": 1,
        "activation_checkpointing": {
            "partition_activations": False,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": False,
            "number_checkpoints": 4,
            "synchronize_checkpoint_boundary": True,
            "profile": False
        }
    }


    modelEngine, optimizer, trainLoader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        optimizer=optimizer,
        config_params=ds_config
    )

    device = get_accelerator().device_name(args.local_rank)
    trainer = DeepSpeedTrainer(modelEngine, trainLoader, validation_data, optimizer, args.local_rank, save_every, snapshot_path, args.batch_size, profiler)
    
    setMonitorStart()
    trainer.train(total_epochs)
    if(monitorCheck): 
        monitor.printTotalTrainingTime()
        monitor.printEndTime()


if __name__ == "__main__":
    main()