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
            self.state = np.random.randint(self.nS)
        else:
            self.state = 0
        return self.state
    def act(self, action):
        terminal = False
        if self.state == self.final_state:
            if action == 1:#not-repair
                reward = -np.random.normal(100, 800)
                terminal = True
            elif action == 0:
                reward = -np.random.normal(130,20)
                terminal = True
            else:
                raise Exception("undefined action")
            self.state = 0
        else:
            if action == 1:
                reward = -np.random.normal(0, 1e-4)
                self.state += 1
            elif action == 0:
                reward = -np.random.normal(130, 1)
                terminal = True
                self.state = 0
            else:
                raise Exception("undefined action")
        return self.state, reward, terminal
