# Config Files
class Config():
    def __init__(self, env):
        # General config
        self.resizeImageSize = 84
        self.nActions = env.action_space.n


        # ACP
        self.acpBatchSize = 32
        self.acpMemorySize = int(1e6)
        self.acpNStates = 2 # stacked number of frames
        self.acpLrStart = 1e-3 # initial learning rate
        self.acpLrDecayStep = 1e5 # Final learning rate
        self.acpLrDecayRate = 0.99 # schedule

