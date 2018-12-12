import argparse

import os
import numpy as np
from env import ENV
from MRL import hindsight, outcome, mbie
import time
import sys
import matplotlib.pyplot as plt

def parse_args():
     parser = argparse.ArgumentParser()
     parser.add_argument('--method', default='MBIE', help='what method to use')
     parser.add_argument('--ent_known', type=int, default=0, help='if knowing entropy')
     parser.add_argument('--random', type=float, default=0, help='how much randomness to add')
     parser.add_argument('--map_name', default='maps/safe.txt', help='map name')
     parser.add_argument('--n_trial', type=int, default=2, help='number of trail for each method')
     parser.add_argument('--n_sample', type=int, default=1, help='number of samples for random env')
     parser.add_argument('--max_ep', type=int, default=500, help = 'maximum number of episdoes')
     parser.add_argument('--max_step', type=int, default=1000, help='maximum number of steps')
     parser.add_argument('--beta', type=float, default=2, help='constant in MBIE')
     parser.add_argument('--lambd', type=float, default=100, help='entropy weight')
     args = parser.parse_args()
     return args

def make_saving_dir(args):
    dictionary = vars(args)
    string = "./results/"
    for key in dictionary.keys():
        string += "{}_{}/".format(key, dictionary[key])
    if not os.path.exists(os.path.dirname(string)):
        os.makedirs(os.path.dirname(string))
    string += "results_"
    return string
def disc_return(R, gamma):
    dR = 0
    for i in reversed(range(len(R))):
            dR = gamma * dR + R[i]
    return dR

def main():
    args = parse_args()
    for key in vars(args).keys():
        print('[*] {} = {}'.format(key, vars(args)[key]))

    save_dir = make_saving_dir(args)
    print(save_dir)
    result = np.zeros((args.n_sample, args.n_trial, args.max_ep, 2))

    for sample in range(args.n_sample):
        env = ENV(mapFile=args.map_name, random=args.random)
        model = {'MBIE': mbie.MBIE(env, args.beta), 'MBIE_NS': mbie.MBIE_NS(env, args.beta),\
            'DH': hindsight.DH(env, bool(args.ent_known),args.beta, args.lambd),\
            'DO': outcome.DO(env, bool(args.ent_known), args.beta, args.lambd)}
        print('sample {} out of {}'.format(sample, args.n_sample))
        env._render()

        np.save(save_dir + "map_sample_{}.npy".format(sample), env.map)
        for trial in range(args.n_trial):
            print('trail = {}'.format(trial))
            mrl = model[args.method]
            mrl.reset()
            for episode in range(args.max_ep):
                terminal = False
                step = 0
                R = []
                s = env.reset()
                while not terminal and step < args.max_step:
                    action = np.random.choice(np.flatnonzero(mrl.Q[s, :] == mrl.Q[s,:].max()))
                    ns, r, terminal = env.step(action)
                    ent = 0.1 * 1/(1 + 10*np.mean(mrl.entropy[s, ns, :]))
                    r += ent
                    R.append(r)
                    mrl.observe(s,action,ns,r, terminal)
                    step += 1
                    s = ns
                    #print(step)
                    env._render()

                result[sample, trial, episode, 0] = step
                result[sample, trial, episode, 1] = disc_return(R, mrl.gamma)
                mrl.Qupdate()
                print(episode, step, disc_return(R, mrl.gamma), np.max(mrl.Q))
                print(np.argmax(mrl.Q, axis=1).reshape(11,11))
                env._render()
            try:
                np.save(save_dir + "entopy_trail_{}_sample_{}.npy".format(trial, sample), mrl.entropy)
            except:
                print("No entropy is saving")
            np.save(save_dir + "count_trail_{}_sample_{}.npy".format(trial, sample), mrl.count)
    np.save(save_dir + 'results.npy', result)
    #plt.plot(np.mean(result[0, :, :, 1], axis = 0))
    #plt.show()
if __name__ == '__main__':
    main()
