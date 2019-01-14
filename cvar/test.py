import numpy as np
import matplotlib.pyplot as plt
from env import mrp
from td import *
from cvar import *
from config import *

def compute_loss(cvar, env):
    it = 0; max_it = 10
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
            s = (s - r)/config.gamma
            r_ep.append(r)
        it += 1
        value = 0
        for r in reversed(r_ep):
            value = r + config.gamma * value
        value_all.append(value)
    return np.mean(value_all)

# ret = [7.591350160376821, 1.0, 26.749176668620702, 4.711430844598082, 15.942525975663852, 1.24255, 1.6350000000000002, 13.575862198712107, 4.711430844598081, 1.9419999999999997]
# plt.hist(ret)
# plt.show()
# exit(0)
env = stopping_env()
config = config(env.nS, env.nA)
ret = np.zeros(10)

for i in range(10):
	cvar = CVaROptimize(config)
	k = 0
	s = 0
	env.reset()
	x = env.state
	while k < 100000:
	    action = cvar.act(x, s)
	    nx, r, terminal = env.act(action)
	    s = cvar.observe(x, s, action, nx, r, k, terminal)
	    #print(cvar.policy)
	    k += 1
	    x = nx
	    if terminal:
	        env.reset()
	        x = env.state
	ret[i] = print(compute_loss(cvar, env))
print(ret)
plt.hist(ret)
plt.show()