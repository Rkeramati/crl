import six
from termcolor import colored
import sys
import numpy as np

class Gridworld:
    def __init__(self, mapFile='map.txt', random = 0, random_action_prob=0.1):

        self.to_int = {'#': 0, ' ':1, 's': 2, 'g': 3, '0': 4, 'j': 5}
        self.to_symbol = {v: k for k, v in self.to_int.items()}
        self.random_symbol = 's'
        self.random_prob = random_action_prob

        self.reward_map = {'#': 0, ' ':0, 's': 0, 'g': 1, '0': 0, 'j':-0.3}
        self.terminal_map = {'#': 0,  ' ':0, 's': 0, 'g': 1, '0': 0, 'j':0}
        self.action_space = ['UP', 'RI', 'DO', 'LE']
        self.nA = len(self.action_space)
        self.reward_step = -0.15

        self.map, self.state, self.reward, self.terminal = self.readmap(mapFile, random)

        #self.reset(random=False)
        row, col = np.where(self.map == 4)
        self.initial_state = (row[0], col[0])
        self.current_state = self.initial_state
        self.current_terminal = False

    def reset(self, random=False):
        # Random intialization
        if random:
            picked = False
            while not picked:
                row = np.random.randint(self.size)
                col = np.random.randint(self.size)
                if self.map[row, col] not in [self.to_int['#'], self.to_int['g']]:
                    picked = True

            self.current_state = (row, col)
            self.current_terminal = False
        else:
            self.current_state = self.initial_state
            self.current_terminal = False
            row, col = self.current_state
        return self.state[row, col]

    def step(self, action):

        row_t, col_t = self.current_state
        row, col = self.current_state

        if self.map[row_t, col_t] == self.to_int[self.random_symbol]: #pick action in random
            if np.random.rand() <= self.random_prob:
                action = np.random.randint(self.nA)

        if self.action_space[action] == 'RI':
            col_t += 1
        if self.action_space[action] == 'LE':
            col_t -= 1
        if self.action_space[action] == 'UP':
            row_t -= 1
        if self.action_space[action] == 'DO':
            row_t += 1
        # Check if valid
        if self.map[row_t, col_t] != self.to_int['#'] and not self.current_terminal:
            row, col = row_t, col_t

        reward = self.reward[row, col]
        if self.current_terminal:
            reward = 0

        self.current_state = (row, col)
        self.current_terminal = bool(self.terminal[row, col])


        return self.state[row, col], reward+self.reward_step, self.terminal[row, col]


    def readmap(self, mapFile, random=0):
        f = open(mapFile, 'r')
        counter = 1
        for line in f:
            if line[0] == '$': #info line
                size = int(line[1:])
                self.size = size
                state = np.zeros((size, size), dtype=np.int32)
                info = np.zeros((size, size))
                rew = np.zeros((size, size))
                terminal = np.zeros((size, size))
                row = 0
                continue
            for col in range(size):
                state[row, col] = counter
                symbol = line[col]
                if symbol not in ['#', 'g']:
                    if np.random.rand() < random:
                        symbol = self.random_symbol

                info[row, col] = self.to_int[symbol]
                rew[row, col] = self.reward_map[line[col]]
                terminal[row, col] = self.terminal_map[line[col]]
                counter += 1

            row+=1
        self.nS = counter - 1 #number of states

        return info, state, rew, terminal

    def _render(self):
        size = self.map.shape[0]
        desc = [[self.to_symbol[self.map[y, x]]\
                for x in range(size)] for y in range(size)]
        row , col = self.current_state
        desc[row][col] = 'X'
        print("\n".join(''.join(line) for line in desc)+"\n")
