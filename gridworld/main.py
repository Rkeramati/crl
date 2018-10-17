import numpy as np
from fourroom import env
import time
import sys

test = env()
test._render()

action = [1, 1, 1, 1, 2, 2, 1, 1, 1 ,1,\
        2, 2 ,2 ,2 ,2 ,2]
for i in range(len(action)):
    #action = np.random.randint(4)
    print(test.step(action[i]))
    test._render()
    time.sleep(1)
