import numpy as np
import matplotlib.pyplot as plt
import gym
from env import Gridworld
from mrp import machine_repair

class Config():
    def __init__(self, nS, nA):
        self.Vmin = -0.1
        self.Vmax = 1.1
        self.nAtoms = 51
        self.nS = nS
        self.nA = nA
        self.gamma = 0.99
        self.max_e = 0.9
        self.min_e = 0.1
        self.max_alpha = 0.9
        self.min_alpha = 0.1
        self.episode_ratio = 2

class C51():
    def __init__(self, config, init='uniform', ifCVaR = False, temp=10):
        self.ifCVaR = ifCVaR
        self.config = config
        self.init = init
        self.temp = temp

        if init == 'optimistic':
            self.p = np.zeros((self.config.nS, self.config.nA,\
                        self.config.nAtoms))
            self.p[:, :, -2] = 1
        elif init == 'uniform':
            self.p = np.ones((self.config.nS, self.config.nA,\
                        self.config.nAtoms)) * 1.0/self.config.nAtoms
        elif init == 'random':
            self.p = np.random.rand(self.config.nS, self.config.nA,\
                        self.config.nAtoms)
            for x in range(self.config.nS):
                for a in range(self.config.nA):
                    self.p[x, a, :] /= np.sum(self.p[x, a, :])
        elif init =='temp':
            self.p = np.random.rand(self.config.nS, self.config.nA,\
                        self.config.nAtoms)
            for x in range(self.config.nS):
                for a in range(self.config.nA):
                    self.p[x, a, :] = self.softmax(self.p[x, a, :], temp=temp)
        else:
            self.p = np.ones((self.config.nS, self.config.nA,\
                        self.config.nAtoms)) * 1.0/self.config.nAtoms
            self.p[:, :, 0] = 10
            for x in range(self.config.nS):
                for a in range(self.config.nA):
                    self.p[x, a, :] /= np.sum(self.p[x, a, :])

        self.dz = (self.config.Vmax - self.config.Vmin)/(self.config.nAtoms-1)
        self.z = np.arange(self.config.nAtoms) * self.dz + self.config.Vmin
    def observe(self, x, a, r, nx, terminal, alpha):
        if not self.ifCVaR: #Normal
            Q_nx = np.zeros(self.config.nA)
            for at in range(self.config.nA):
                Q_nx[at] = np.sum(self.p[nx, at, :] * self.z)
            a_star = np.argmax(Q_nx)
        else:
            Q_nx = self.CVaR(x, alpha, N=20)
            a_star = np.argmax(Q_nx)

        m = np.zeros(self.config.nAtoms)
        for i in range(self.config.nAtoms):
            if not terminal:
                tz = np.clip(r + self.config.gamma*self.z[i],\
                        self.config.Vmin, self.config.Vmax)
            else:
                tz = np.clip(r,\
                        self.config.Vmin, self.config.Vmax)
            b = (tz - self.config.Vmin)/self.dz
            l = int(np.floor(b)); u = int(np.ceil(b))
            m[l] += self.p[nx, a_star,i] * (u-b)
            m[u] += self.p[nx, a_star,i] * (b-l)
        self.p[x, a, :] = self.p[x, a, :] + alpha * (m - self.p[x, a, :])
        if self.init == 'temp':
            self.p[x, a, :] = self.softmax(self.p[x, a, :], self.temp)
        else:
            self.p[x, a, :] /= np.sum(self.p[x, a, :])

    def Q(self, x):
        Q_nx = np.zeros(self.config.nA)
        for at in range(self.config.nA):
            Q_nx[at] = np.sum(self.p[x, at, :] * self.z)
        return Q_nx

    def CVaR(self, x, alpha, N=20):
        Q = np.zeros(self.config.nA)
        for a in range(self.config.nA):
            values = np.zeros(N)
            for n in range(N):
                tau = np.random.uniform(0, alpha)
                idx = np.argmax((np.cumsum(self.p[x, a, :]) > tau) * 1.0)
                z = self.z[idx]
                values[n] = z
            Q[a] = np.mean(values)
        return Q
    def softmax(self, x, temp):
        e_x = np.exp((x - np.max(x))/temp)
        return e_x / e_x.sum()

def run(k):
    world = machine_repair()

    config = Config(world.nS, world.nA)
    config.Vmin = -30; config.Vmax = 30
    c51 = C51(config, init = 'random', ifCVaR = True)
    counts = np.zeros((world.nS, world.nA)) + 1

    const = 20
    num_episode = 2000
    trial = 100
    returns = np.zeros((num_episode, trial))
    returns_online = np.zeros((num_episode, trial))

    CVaRs = np.zeros((num_episode, world.nA))
    for ep in range(num_episode):
        terminal = False
        alpha = max(0.001, 0.5 + ep * ((0.4 - 0.5)/(num_episode/2)))
        o = world.reset()
        o_init = o
        ret = 0
        while not terminal:
            a = np.argmax(c51.CVaR(o, alpha=0.5, N=50) + const/np.sqrt(counts[o, :]))
            no, r, terminal = world.step(a)
            counts[o, a] += 1
            ret += r
            c51.observe(o, a, r + const/np.sqrt(counts[o, a]), no, terminal, alpha)
            o = no

        tot_rep = np.zeros(trial)
        for ep_t in range(trial):
            terminal = False
            o = world.reset()
            o_init = o
            ret = 0
            while not terminal:
                a = np.argmax(c51.CVaR(o, alpha=0.5, N=50))
                no, r, terminal = world.step(a)
                ret += r
                o = no
            tot_rep[ep_t] = ret
        returns[ep, :] = tot_rep

        tot_rep = np.zeros(trial)
        for ep_t in range(trial):
            terminal = False
            o = world.reset()
            o_init = o
            ret = 0
            while not terminal:
                a = np.argmax(c51.CVaR(o, alpha=0.5, N=50) + const/np.sqrt(counts[o, :]))
                no, r, terminal = world.step(a)
                ret += r
                o = no
            tot_rep[ep_t] = ret
        returns_online[ep, :] = tot_rep
        if ep%100 == 0:
            print('episode: %d'%(ep))
    np.save('count_based_online_%d.npy'%(k), returns_online)
    np.save('count_based_eval_%d.npy'%(k), returns)

for i in range(10):
    print('Trail: %d out of %d'%(i, 10))
    run(i)
