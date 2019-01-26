import numpy as np
from env import mrp
from pg import *
from td import *
from config import *


def test_policy(PG, trial=50):
    env = mrp.machine_repair()
    state = env.reset()
    totalReturn = np.zeros(trial)
    for iteration in range(trial):
        terminal = False
        ret = []
        while not terminal:
            policy = PG.policy(state)
            #action = np.random.choice(np.arange(env.nA), p = policy)
            action = np.argmax(policy)
            next_state, reward, terminal = env.act(action)
            ret.append(reward)
            state = next_state
        state = env.reset()
        returnEp = 0
        for r in reversed(ret):
            returnEp = returnEp + r
        totalReturn[iteration] = returnEp
    return np.mean(totalReturn), np.var(totalReturn)

def main(verbose):
    env = mrp.machine_repair()
    const = config(env.nS, env.nA, 'mrp')
    PG = pg(const)
    TD = TDVar(const)

    state = env.reset()
    for ep in range(const.numIteration):
        terminal = False
        episode = []
        while not terminal:
            policy = PG.policy(state)
            action = np.random.choice(np.arange(env.nA), p = policy)
            next_state, reward, terminal = env.act(action)
            episode.append((state, action, reward, next_state, terminal))
            state = next_state
        state = env.reset()
        ep += 1
        TD.observe(episode)
        PG.observe(episode)
    print('## Final Poicy ##')
    for s in range(env.nS):
        print('state %s, value %g, action %d'%(s, np.max(PG.policy(s)), np.argmax(PG.policy(s))))
    print('Avergae return %g, variance %g'%(test_policy(PG)))

if __name__ == '__main__':
    main(True)
