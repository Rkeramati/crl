import numpy as np
import matplotlib.pyplot as plt
# Defining a machine repair class, for machine repair env
# terminal State is a recuring state
class Nav2D():
    def __init__(self, seed=6):
        np.random.seed(seed)

        self.action_space = ["UP", "RI", "DO", "LE"]
        self.nA = len(self.action_space)

        self.maxX = 64
        self.maxY = 53
        self.nS = self.maxX * self.maxY

        self.gamma = 0.95
        self.delta = 0.05
        self.M = 2/(1-self.gamma)
        self.initial_state = (60, 50)
        self.goal_state = (60, 2)

        self.obstacles = []
        exp = 0.25
        p = np.zeros((self.maxX, self.maxY))
        for i in range(0, 62):
            px = np.exp((i/62)**(exp))/np.exp(1) * 0.3
            print(px)
            for j in range(10, 45):
                py = 1-np.exp((abs((j-10) - 17.5)/17.5)**(exp))/np.exp(1) * 0.3
                p[i,j] = px * py * np.random.randn()
        idx = np.argsort(-p.flatten())[0:80]

        prob = np.zeros((self.maxX, self.maxY)).flatten()
        prob[idx] = 1
        self.obstacles = prob.reshape(((self.maxX, self.maxY)))

        self.terminal = False
        self.current_state = self.initial_state

    def idx(self, state):
        x, y = state
        return x * self.maxX + y

    def reset(self):
        self.current_state = self.initial_state
        self.terminal = False
        return self.idx(self.current_state)

    def step(self, action):
        x, y =self.current_state
        if self.terminal:
            return self.idx(self.current_state)
        reward = -1
        # Move
        if self.action_space[action] == "RI":
            x+=1
        if self.action_space[action] == "LE":
            x-=1
        if self.action_space[action] == "DO":
            y-=1
        if self.action_space[action] == "UP":
            y+=1

        #check Wall:
        if x>= self.maxX:
            x = self.maxX
        if x<=0:
            x = 0
        if y>=self.maxY:
            y=self.maxY
        if y<=0:
            y=0

        self.current_state = (x, y)
        # Checl Obstacle
        if self.obstacles[x, y] == 1:
            reward += -self.M
            self.terminal = True

        #Check goal:
        if self.current_state == self.goal_state:
            reward = 0
            self.terminal = True

        return self.idx(self.current_state), reward, self.terminal

    def _render(self):
        p = self.obstacles
        x, y = self.current_state
        p[x, y] = 2
        plt.matshow(p.T)
        plt.show()

