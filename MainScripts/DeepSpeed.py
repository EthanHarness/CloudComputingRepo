import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
import psutil
from datetime import timedelta
import random

from Imagenet1kDataset import CustomImageNet1000
from Inference import Inference
from PerformanceMonitor import PerformanceMonitor
from ResnetModel import ActivationCheckpointingResnetModel

#TRAIN_SIZE = 131072
TRAIN_SIZE = 2048
VALIDATION_SIZE = -1
PERFORMANCE_FLAG = True
MEMORY_PROFILING_FLAG = False
ENABLE_SAVING = False
RUN_VALIDATIONS = True
monitor = PerformanceMonitor("DeepSpeed")

def deepspeedSetup(rank: int):
    deepspeed.init_distributed(timeout=timedelta(seconds=5400))
    get_accelerator().set_device(rank)

    pid = psutil.Process(os.getpid())
    print(f"Process {pid}")
    
def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=100,
        pin_memory=True,
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
        if os.path.exists(snapshot_path) and ENABLE_SAVING:
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
            
        self.monitorCheck = PERFORMANCE_FLAG and self.gpu_id == 0
        self.profilingCheck = MEMORY_PROFILING_FLAG and self.gpu_id == 0
        
    def setMonitorStart(self):
        if self.monitorCheck: monitor.setPerfStartTime()


    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        path, clientState = self.model.load_checkpoint(snapshot_path)
        self.epochs_run = clientState["EPOCHS_RUN"]
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
        self.model.save_checkpoint("../model", client_state={"EPOCHS_RUN": epoch}, save_latest=True)
        if self.gpu_id == 0: print(f"Epoch {epoch} | Training snapshot saved")

    def train(self, max_epochs: int):
        if (self.monitorCheck): monitor.printTrainTimeStart()
        
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if ENABLE_SAVING and epoch % self.save_every == 0: self._save_snapshot(epoch)
            if (epoch % self.save_every == 0 and RUN_VALIDATIONS): self.runValidations(epoch)
                
            if self.profilingCheck and epoch < monitor.getProfilerSteps(): self.profiler.step()
            if self.profilingCheck and epoch == monitor.getProfilerSteps() - 1:  self.profiler.stop()
            
        if (self.monitorCheck): 
            monitor.printTrainTimeEnd()
            monitor.flushOutput()
        if ENABLE_SAVING and epoch % self.save_every != 0: self._save_snapshot(epoch-1)

    def runValidations(self, epoch):
        self.setMonitorStart()
        outputTensor = self.inf.runValidations("dp", self.model, self.validation_data, self.gpu_id)
        if (self.monitorCheck): monitor.printValidationTime(epoch)
        outputTensor = outputTensor.to(f'cuda:{self.gpu_id}')
        
        if self.gpu_id == 0:
            gathered_data = [torch.zeros_like(outputTensor) for _ in range(dist.get_world_size())]
            dist.gather(outputTensor, gather_list=gathered_data, dst=0)
            gathered_data = torch.sum(torch.stack(gathered_data, dim=0), dim=0)
            print(f"Validation Results: {gathered_data[0].item()}/{gathered_data[1].item()}={gathered_data[0].item()/gathered_data[1].item()*100}%")
        else:
            dist.gather(outputTensor, dst=0)
        
def load_train_objs():
    train_set = CustomImageNet1000("train", False, TRAIN_SIZE)
    valid_set = CustomImageNet1000("validation", False, VALIDATION_SIZE)
    model = ActivationCheckpointingResnetModel().createModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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
    parser.add_argument('--snapshot_path', default="../model/", type=str, help='Model snapshot save location')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    save_every = args.save_every
    total_epochs = args.total_epochs
    snapshot_path = args.snapshot_path + "snapshot_DeepSpeed.pt"

    print(f"WORLD_SIZE: {torch.cuda.device_count()} RANK {args.local_rank}")
    if args.local_rank == 0:
        print(f"Number of available GPUs: {world_size}")
        for i in range(world_size):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

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
            "overlap_comm": True,
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
    
    if args.local_rank == 0:
        objList = [random.randint(1, 100000000), random.randint(1, 100000000)]
    else:
        objList = [None, None]
    torch.distributed.broadcast_object_list(objList, src=0)
    seed1 = objList[0]
    seed2 = objList[1]
    dataset.setSeed(seed1)
    validation_data.setSeed(seed2)

    trainer = DeepSpeedTrainer(modelEngine, trainLoader, validation_data, optimizer, args.local_rank, save_every, snapshot_path, args.batch_size, profiler)
    
    setMonitorStart()
    trainer.train(total_epochs)
    if(monitorCheck): 
        monitor.printTotalTrainingTime()
        monitor.printEndTime()
    if (MEMORY_PROFILING_FLAG and args.local_rank == 0):
        torch.cuda.synchronize()
        monitor.exportMemory(profiler)


if __name__ == "__main__":
    main()