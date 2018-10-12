import acp.acpBrain as Brain
import acp.acpMemory as Memory
import acp.acpVision as Vision
import numpy as np

class acp():
    def __init__(self, config):
        self.brain = Brain.acpBrain(config)
        self.memory = Memory.acpMemory(self.brain.getInputSize(),\
                self.brain.getLabelSize(), config)
        self.vision = Vision.acpVision()
        self.observation = []
        self.nObs = config.acpNStates
        self.nA = config.nActions

    def makeInputLabel(self):
        # makes the observation into input output shape

        # last action, one-hot vector
        _, lastAction, _ = self.observation[-1]
        nnLabel = np.zeros(self.nA)
        nnLabel[lastAction] = 1

        # inialize the size
        nnInputSize = self.brain.getInputSize()
        nnInputSize.pop(0) # First is None!
        nnInput =np.zeros(tuple(nnInputSize))

        # now add all ns in the tuples in observation
        for idx in reversed(range(len(self.observation))):
            _, _, s = self.observation[idx]
            # cahnnel is the third dimension
            nnInput[:, :, idx] = s

        return nnInput, nnLabel


    def observe(self, sess, s, a, ns):
        # sess: tf session, state: s, action: a, next state: ns tuple
        # preprocess both s and ns:
        s = self.vision.process(s)
        ns = self.vision.process(ns)

        if len(self.observation) >= self.nObs:
            self.observation.pop(0)
            self.observation.append((s, a, ns))
        else:
            self.observation.append((s, a, ns))

        nnInput, nnLabel = self.makeInputLabel()
        self.memory.add(nnInput, nnLabel)
        return self.brain.infer(sess, nnInput)

    def train(self, sess):
        nnInput, nnLabel = self.memory.sample()
        return self.brain.train(sess, nnLabel, nnInput)

