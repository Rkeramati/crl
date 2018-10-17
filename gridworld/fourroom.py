import six
from termcolor import colored
import sys
import numpy as np

class env:
    def __init__(self, mapFile='map.txt'):

        self.to_int = {'#': 0, ' ':1, 's': 2, 'g': 3, '0': 4}
        self.to_symbol = {v: k for k, v in self.to_int.items()}

        self.reward_map = {'#': 0, ' ':0, 's': 0, 'g': 1, '0': 0}
        self.terminal_map = {'#': 0,  ' ':0, 's': 0, 'g': 1, '0': 4}
        self.action_space = ['UP', 'RI', 'DO', 'LE']
        self.nA = len(self.action_space)

        self.map, self.state, self.reward, self.terminal = self.readmap(mapFile)

        row, col = np.where(self.map == 4)
        self.initial_state = (row[0], col[0])
        self.current_state = self.initial_state
        self.current_terminal = False

    def reset(self):
        self.current_state = self.initial_state
        self.current_terminal = False

    def step(self, action):

        row, col = self.current_state
        if self.action_space[action] == 'RI':
            col += 1
        if self.action_space[action] == 'LE':
            col -= 1
        if self.action_space[action] == 'UP':
            row -= 1
        if self.action_space[action] == 'DO':
            row += 1
        # Check if valid
        if self.map[row, col] != self.to_int['#'] and not self.current_terminal:
            self.current_state = (row, col)

        row, col = self.current_state
        self.current_terminal = bool(self.terminal[row, col])

        return self.state[row, col], self.reward[row, col], self.terminal[row, col]


    def readmap(self, mapFile):
        f = open(mapFile, 'r')
        counter = 0
        for line in f:
            if line[0] == '$': #info line
                size = int(line[1:])
                state = np.zeros((size, size), dtype=np.int32)
                info = np.zeros((size, size))
                rew = np.zeros((size, size))
                terminal = np.zeros((size, size))
                row = 0
                continue
            for col in range(size):
                state[row, col] = counter
                info[row, col] = self.to_int[line[col]]
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
