import numpy as np
# Defining a machine repair class, for machine repair env
class machine_repair():
    def __init__(self, uniform=False):
        self.nS = 50
        self.nA = 2
        self.uniform = uniform
        self.reset()
        self.final_state = 49
    def reset(self):
        if self.uniform:
            self.s = np.random.randint(self.nS)
        else:
            self.s = 0
        return self.s
    def step(self, action):
        if self.s == self.final_state:
            if action == 1:
                reward = -np.random.normal(100, 800)
            elif action == 0:
                reward = -np.random.normal(130,20)
            else:
                raise Exception("undefined action")
            self.s = 0
        else:
            if action == 1:
                reward = -np.random.normal(0, 1e-4)
                self.s += 1
            elif action == 0:
                reward = -np.random.normal(130, 1)
                self.s = 0
            else:
                raise Exception("undefined action")
        return self.s, reward, False
