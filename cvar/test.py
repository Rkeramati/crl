import numpy as np
import matplotlib.pyplot as plt
from env import mrp, stopping
from td import *
from cvar import *
from config import *

def compute_loss(cvar, env):
    it = 0; max_it = 5
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
            x = nx
        it += 1
        value = 0
        for r in reversed(r_ep):
            value = r + config.gamma * value
        value_all.append(value)
    return np.mean(value_all)

env = stopping.stopping_env()
config = config(env.nS, env.nA)

max_it = 50
max_step = 1000
step_by_step = np.zeros((max_it, max_step))
ret = np.zeros(max_it)
for i in range(max_it):
    cvar = CVaROptimize(config)
    td = TD_learner(config.nS, config.gamma)

    k = 0; s = 0; env.reset()
    x = env.state
    while k < max_step:
        action = cvar.act(x, s)
        nx, r, terminal = env.act(action)
        td.observe(x, r, nx)
        var = td.variance[x]
        #print(var)
        r -= 0.01 * var
        s = cvar.observe(x, s, action, nx, r, k, terminal)
        x = nx
        if terminal:
            env.reset()
            x = env.state;
        step_by_step[i, k] = compute_loss(cvar, env)
        #print(step_by_step[i,k])
        k += 1
    ret[i] = compute_loss(cvar, env)
    np.save('step_by_step_var_0.01.npy', step_by_step)
    np.save('return_var_0.01.npy', ret)
    #print(s)
    print(i, ret[i])
#print(td.variance)
print(ret)
#print(np.mean(ret), np.std(ret))
np.save('step_by_step_var_0.01.npy', step_by_step)
np.save('return_var_0.01.npy', ret)
plt.hist(ret)
plt.show()
