import numpy as np
# Defining a machine repair class, for machine repair env
# terminal State is a recuring state
class machine_repair():
    def __init__(self, uniform=False):
        self.nS = 10
        self.nA = 2
        self.repairRew = 130; self.repairStd = 1
        self.notRepairRew = 0; self.notRepairStd = 1e-4
        self.repairRewEnd = 130; self.repairRewEndStd = 20
        self.notRepairRewEnd = 100; self.notReapirRewEndStd = 800

        self.uniform = uniform
        self.reset()
        self.final_state = self.nS - 2
        self.terminal_state = self.nS - 1

    def reset(self):
        if self.uniform:
            self.state = np.random.randint(self.nS)
        else:
            self.state = 0
        return self.state
    def step(self, action):
        if self.state == self.terminal_state:
            return self.state, 0, True

        terminal = False
        if self.state == self.final_state:
            if action == 1:#not-repair
                reward = -np.random.normal(self.notRepairRewEnd, self.notReapirRewEndStd)
                self.state = self.terminal_state
            elif action == 0:#repair
                reward = -np.random.normal(self.repairRewEnd, self.repairRewEndStd)
                self.state = self.terminal_state
                terminal = True
            else:
                raise Exception("undefined action")

        else:
            if action == 1: #not repair
                reward = -np.random.normal(self.notRepairRew, self.notRepairStd)
                self.state += 1
            elif action == 0: #repair
                reward = -np.random.normal(self.repairRew, self.repairStd)
                self.state = self.terminal_state
                terminal = True
            else:
                raise Exception("undefined action")
        return self.state, reward, terminal
