import time

class PerformanceMonitor:
    def __init__(self, type):
        self.timers = []
        self.startString = f"MONITOR_{type}"
    
    #As long as start time and elapsed are called, should work with each other. 
    def setPerfStartTime(self): self.timers.append(time.perf_counter())
    def getElapsedTime(self): return time.perf_counter() - self.timers.pop()
    
    def printSetupTime(self): print(f"{self.startString}-SETUP-TIME: {self.getElapsedTime()}")
    def printLoadingTrainingTime(self): print(f"{self.startString}-LOADING-TRAINING-OBJECTS-TIME: {self.getElapsedTime()}")
    def printCreatingTrainDataloaderTime(self): print(f"{self.startString}-CREATING-TRAINING-DATALOADER-TIME: {self.getElapsedTime()}")
    def printCreatingValidDataloaderTime(self): print(f"{self.startString}-CREATING-VALIDATION-DATALOADER-TIME: {self.getElapsedTime()}")
    def printCreatingTrainingClass(self): print(f"{self.startString}-CREATING-TRAIN-OBJECT-TIME: {self.getElapsedTime()}")
    def printTotalTrainingTime(self): print(f"{self.startString}-TRAINING-TIME: {self.getElapsedTime()}")
    def printEpochRuntime(self, epoch): print(f"{self.startString}-TRAIN-EPOCH-{epoch}-TIME: {self.getElapsedTime()}")
    def printValidationTime(self,epoch): print(f"{self.startString}-VALIDATION-EPOCH-{epoch}-TIME: {self.getElapsedTime()}")
        
    def printStartTime(self): print(f"{self.startString}-START-TIME: {time.ctime()}")
    def printEndTime(self): print(f"{self.startString}-END-TIME: {time.ctime()}")
    def printTrainTimeStart(self): print(f"{self.startString}-TRAIN-START-TIME: {time.ctime()}")
    def printTrainTimeEnd(self): print(f"{self.startString}-TRAIN-END-TIME: {time.ctime()}")
    