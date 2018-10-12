import numpy as np

class acpMemory():
    # Implement the replay buffer of acp (ACtion Predictor)
    def __init__(self, inputSize, labelSize, config):
        # Args: list inputSize, labelSize, coming from acp.inputSize, acp.labelSize
        self.batchSize = config.acpBatchSize
        self.memorySize = config.acpMemorySize
        self.counter = 0 #Counting number of elements in the memory
        self.memory = [None] * self.memorySize

        # Defining sizes, coming from TF, frist element might be Batch Size = None
        assert inputSize[0]==None, 'inputSize first element should be None, for BatchSize'
        assert labelSize[0]==None, 'labelSize first element should be None, for BatchSize'
        assert len(inputSize) == 4, 'input should be 4 dimension, (B, H, W, C)'
        assert len(labelSize) == 2, 'label should be 2 dimension, (B, L)'

        inputSize[0] = self.batchSize
        labelSize[0] = self.batchSize
        self.inputSize = tuple(inputSize)
        self.labelSize = tuple(labelSize)

    def add(self, nnInput, nnLabel):
        # add a new example, nnInput and nnLabel to the memory
        # class acpPreprocess should make sure the shapes are input are as expected to
        # Neural net
        if self.counter >= self.memorySize:
            self.memory.pop(0) # Remove the least recent example
            self.memory.append((nnInput, nnLabel)) # Assign to the last one
        else:
            self.memory[self.counter] = (nnInput, nnLabel)
            self.counter += 1
        return 0

    def sample(self):
        # sample a batch of label and inputs with batch Size
        idx = np.random.choice(self.counter, self.batchSize)
        nnInput = np.zeros(self.inputSize)
        nnLabel = np.zeros(self.labelSize)
        counter = 0
        for i in idx:
            nnInput[counter, :, :, :], nnLabel[counter, :] = self.memory[i]
            counter += 1

        return nnInput, nnLabel

