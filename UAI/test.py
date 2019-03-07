import numpy as np
import matplotlib.pyplot as plt

from terrain import Nav2D
env = Nav2D()
for i in range(100):
	env.step(np.random.randint(env.nA))
	env._render()