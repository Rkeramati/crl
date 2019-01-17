import numpy as np
import matplotlib.pyplot as plt
from env import mrp
from td import *
from cvar import *
from config import *

def compute_loss(cvar, env):
    it = 0; max_it = 50
    value_all = []
    while it < max_it:
        env.reset()
        x = env.state
        s = 0
        terminal = False
        r_ep = []
        while not terminal:
            action = cvar.act(x, s)
            nx, r, terminal = env.act(action)
            s = (s-r)/config.gamma
            r_ep.append(r)
        it += 1
        value = 0
        for r in reversed(r_ep):
            value = r + config.gamma * value
        value_all.append(value)
        return np.mean(value_all)

env = stopping_env()
config = config(env.nS, env.nA)

max_it = 50
ret = np.zeros(max_it)
for i in range(max_it):
    cvar = CVaROptimize(config)
    k = 0; s = 0; env.reset()
    x = env.state
    while k < 1000:
        action = cvar.act(x, s)
        nx, r, terminal = env.act(action)
        s = cvar.observe(x, s, action, nx, r, k, terminal)
        k += 1
        x = nx
        if terminal:
            env.reset()
            x = env.state;
        #if k%5 == 0:
        #    print('iteration %d'%(k), compute_loss(cvar, env))
    ret[i] = compute_loss(cvar, env)
    print(ret[i])
print(ret)
print(np.mean(ret), np.std(ret))
plt.hist(ret)
plt.show()
