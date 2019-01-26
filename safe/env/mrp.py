import numpy as np
# Defining a machine repair class, for machine repair env
# terminal State is a recuring state
class machine_repair():
    def __init__(self, uniform=False):
        self.nS = 5
        self.nA = 2
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
    def act(self, action):
        if self.state == self.terminal_state:
            return self.state, 0, True

        terminal = False
        if self.state == self.final_state:
            if action == 1:#not-repair
                reward = -np.random.normal(0.8, 1)
                terminal = True
                self.state = self.terminal_state
            elif action == 0:
                reward = -np.random.normal(1,0.1)
                terminal = True
                self.state = self.terminal_state
            else:
                raise Exception("undefined action")

        else:
            if action == 1:
                reward = -np.random.normal(0, 1e-4)
                self.state += 1
            elif action == 0:
                reward = -np.random.normal(1.5 - (self.state/5), 0.001)
                terminal = True
                self.state = self.terminal_state
            else:
                raise Exception("undefined action")
        return self.state, reward, terminal
