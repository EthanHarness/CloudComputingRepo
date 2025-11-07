import time
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
import sys
import torch
import os
import shutil

class PerformanceMonitor:
    def __init__(self, type):
        self.timers = []
        self.startString = f"MONITOR_{type}"
        self.memorySnapshotPath = os.path.expanduser(f"~/scratch/memorySnapshots/{type}/")
    
    #As long as start time and elapsed are called, should work with each other. 
    def setPerfStartTime(self): self.timers.append(time.perf_counter())
    def getElapsedTime(self): return time.perf_counter() - self.timers.pop()
    
    def flushOutput(self): sys.stdout.flush()
    
    def printSetupTime(self): print(f"{self.startString}-SETUP-TIME: {self.getElapsedTime()}")
    def printLoadingTrainingTime(self): print(f"{self.startString}-LOADING-TRAINING-OBJECTS-TIME: {self.getElapsedTime()}")
    def printCreatingTrainingClass(self): print(f"{self.startString}-CREATING-TRAIN-OBJECT-TIME: {self.getElapsedTime()}")
    def printTotalTrainingTime(self): print(f"{self.startString}-TRAINING-TIME: {self.getElapsedTime()}")
    def printEpochRuntime(self, epoch): print(f"{self.startString}-TRAIN-EPOCH-{epoch}-TIME: {self.getElapsedTime()}")
    def printValidationTime(self,epoch): print(f"{self.startString}-VALIDATION-EPOCH-{epoch}-TIME: {self.getElapsedTime()}")
        
    def createProfiler(self, gpu_id):
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=5),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{self.memorySnapshotPath}/{gpu_id}/"),
            profile_memory=True,
            record_shapes=True,
            with_stack=True
        )
        profiler.start()
        return profiler
    def exportMemory(self, gpu_id, profiler):
        profiler.export_memory_timeline(f"{self.memorySnapshotPath}/{gpu_id}/Memory.html")

        
    def printStartTime(self): print(f"{self.startString}-START-TIME: {time.ctime()}")
    def printEndTime(self): print(f"{self.startString}-END-TIME: {time.ctime()}")
    def printTrainTimeStart(self): print(f"{self.startString}-TRAIN-START-TIME: {time.ctime()}")
    def printTrainTimeEnd(self): print(f"{self.startString}-TRAIN-END-TIME: {time.ctime()}")
    