import numpy as np
import matplotlib.pyplot as plt

from env import mrp
from pg import *
from td import *
from config import *

import argparse
parser = argparse.ArgumentParser(description='Variance Optimization')
parser.add_argument('name')
parser.add_argument('lambd')

args = parser.parse_args()

def test_policy(PG, trial=20):
    env = mrp.machine_repair()
    state = env.reset()
    totalReturn = np.zeros(trial)
    for iteration in range(trial):
        terminal = False
        ret = []
        while not terminal:
            policy = PG.policy(state)
            action = np.random.choice(np.arange(env.nA), p = policy)
            #action = np.argmax(policy)
            next_state, reward, terminal = env.act(action)
            ret.append(reward)
            state = next_state
        state = env.reset()
        returnEp = 0
        for r in reversed(ret):
            returnEp = returnEp + r
        totalReturn[iteration] = returnEp
    return np.mean(totalReturn), np.var(totalReturn)

def main(name, lambd):
    env = mrp.machine_repair()
    const = config(env.nS, env.nA, 'mrp')
    PG = pg(const)
    TD = TDVar(const)

    state = env.reset()
    learningCurveValue = np.zeros(const.numIteration)
    learningCurveVariance = np.zeros(const.numIteration)
    stateCount = np.zeros(env.nS)
    TDVariance = np.zeros((env.nS, const.numIteration))

    for ep in range(const.numIteration):
        terminal = False
        episode = []
        episodePG = []
        while not terminal:
            policy = PG.policy(state)
            action = np.random.choice(np.arange(env.nA), p = policy)
            next_state, reward, terminal = env.act(action)
            episode.append((state, action, reward, next_state, terminal))
            episodePG.append((state, action, reward+(lambd*TD.variance(state)/(np.sqrt(stateCount[state]+1))), next_state, terminal))
            stateCount[state] += 1
            state = next_state
        learningCurveValue[ep], learningCurveVariance[ep] = test_policy(PG)
        state = env.reset()
        TD.observe(episode)
        PG.observe(episodePG)
        for s in range(env.nS):
            TDVariance[s, ep] = TD.variance(s)
    print('## Final Poicy ##')
    for s in range(env.nS):
        print('state %s, value %g, action %d'%(s, np.max(PG.policy(s)), np.argmax(PG.policy(s))))
    print('Avergae return %g, variance %g'%(test_policy(PG)))
    np.save('result/variance_%s.npy'%(name), learningCurveVariance)
    np.save('result/value_%s.npy'%(name), learningCurveValue)
    np.save('result/variance_all_%s'%(name), TDVariance)
    #plt.figure()
    #plt.plot(learningCurveValue)
    #plt.figure()
    #plt.plot(learningCurveVariance)
    #plt.show()
if __name__ == '__main__':
    for it in range(3):
        main(args.name+'_%d'%(it), float(args.lambd))
