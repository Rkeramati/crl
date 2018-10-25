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
     parser.add_argument('--map_name', default='maps/map_fourroom_exp1.txt', help='map name')
     parser.add_argument('--n_trial', type=int, default=10, help='number of trail for each method')
     parser.add_argument('--n_sample', type=int, default=1, help='number of samples for random env')
     parser.add_argument('--max_ep', type=int, default=2000, help = 'maximum number of episdoes')
     parser.add_argument('--max_step', type=int, default=1000, help='maximum number of steps')
     parser.add_argument('--beta', type=float, default=2, help='constant in MBIE')
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
    for key in vars(args).keys():
        print('[*] {} = {}'.format(key, vars(args)[key]))

    save_dir = make_saving_dir(args)
    result = np.zeros((args.n_sample, args.n_trial, args.max_ep))

    for sample in range(args.n_sample):
        env = ENV(mapFile=args.map_name, random=args.random)
        model = {'MBIE': mbie.MBIE(env, args.beta), 'MBIE_NS': mbie.MBIE_NS(env, args.beta),\
            'DH': hindsight.DH(env, bool(args.ent_known),args.beta),\
            'DO': outcome.DO(env, bool(args.ent_known), args.beta)}
        print('sample {} out of {}'.format(sample, args.n_sample))
        env._render()

        np.save(save_dir + "map_sample_{}.npy".format(sample), env.map)
        for trial in range(args.n_trial):
            mrl = model[args.method]
            mrl.reset()
            for episode in range(args.max_ep):
                terminal = False
                step = 0
                s = env.reset()
                while not terminal and step < args.max_step:
                    action = np.random.choice(np.flatnonzero(mrl.Q[s, :] == mrl.Q[s,:].max()))
                    ns, r, terminal = env.step(action)
                    mrl.observe(s,action,ns,r)
                    step += 1
                    s = ns
                result[sample, trial, episode] = step
                mrl.Qupdate()
                print(episode, step, np.max(mrl.Q))
                #print(np.max(mrl.Q, axis=1).reshape(13,13))
            try:
                np.save(save_dir + "entopy_trail_{}_sample_{}.npy".format(trial, sample), mrl.entropy)
            except:
                print("No entropy is saving")
            np.save(save_dir + "count_trail_{}_sample_{}.npy".format(trial, sample), mrl.count)
    np.save(save_dir + 'results.npy', result)
    #plt.plot(np.mean(result[0, :, :], axis = 0))
    #plt.show()
if __name__ == '__main__':
    main()
