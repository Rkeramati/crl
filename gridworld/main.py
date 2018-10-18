import argparse

import os
import numpy as np
from fourroom import fourroom
from modelbased import MRL
import time
import sys
import matplotlib.pyplot as plt

def parse_args():
     parser = argparse.ArgumentParser()
     parser.add_argument('--entropy', type=bool, default=False, help='either to use entropy methods')
     parser.add_argument('--random', type=float, default=0, help='how much randomness to add')
     parser.add_argument('--n_trial', type=int, default=10, help='number of trail for each method')
     parser.add_argument('--n_sample', type=int, default=5, help='number of samples for random env')
     parser.add_argument('--max_ep', type=int, default=2000, help = 'maximum number of episdoes')
     parser.add_argument('--max_step', type=int, default=1000, help='maximum number of steps')
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

def main():
    args = parse_args()
    save_dir = make_saving_dir(args)
    result = np.zeros((args.n_sample, args.n_trial, args.max_ep))

    for sample in range(args.n_sample):
        env = fourroom(random=args.random)
        print('sample {} out of {}'.format(sample, args.n_sample))
        np.save(save_dir + "map_sample_{}.npy".format(sample), env.map)
        for trial in range(args.n_trial):
            mrl = MRL(env.nS, env.nA, entropy=args.entropy)
            for episode in range(args.max_ep):
                terminal = False
                step = 0
                s = env.reset()
                while not terminal and step < args.max_step:
                    action = np.argmax(mrl.Q[s, :])
                    ns, r, terminal = env.step(action)
                    mrl.observe(s,action,ns,r)
                    step += 1
                    s = ns
                result[sample, trial, episode] = step
                mrl.Qupdate()
            np.save(save_dir + "entopy_trail_{}_sample_{}.npy".format(trial, sample), mrl.entropy)
            np.save(save_dir + "count_trail_{}_sample_{}.npy".format(trial, sample), mrl.count)

if __name__ == '__main__':
    main()
