import numpy as np

class DH():
    # Class for performing model based RL
    # reward being a function of s, a, ns
    def __init__(self, env, entropy_known=False, const=2, lambd=10):
        self.entropy_known = entropy_known
        self.env = env

        #self.printInfo()
        self.gamma = 0.99
        self.beta = const
        self.lambd = lambd
        self._updated = False
        self.it = 20 # iteration for solving

        self.nS = env.nS # number of states
        self.nA = env.nA # number of actions
        self.reset()
    def reset(self):
        self.count = np.zeros((self.nS, self.nS, self.nA))+1 # Counting s,a pairs
        self.reward = np.zeros((self.nS, self.nS, self.nA)) # Reward Only function of states
        self.transitions = np.zeros((self.nS, self.nS, self.nA)) # s, s', a
        self.Q = np.zeros((self.nS, self.nA))
        self.entropy = np.zeros((self.nS, self.nS, self.nA)) + np.log(self.nA) # For all action entropy is equal
        self.terminal = np.zeros(self.nS)

        if self.entropy_known:
            self.fill_entropy()

        self._total_reward = np.zeros((self.nS, self.nS, self.nA))
        self._transition_count = np.zeros((self.nS, self.nS, self.nA)) + 1

    def observe(self, s, a, ns, r, terminal):
        # adding state action next state reward to the history
        # reward is associated with ns
        self._updated = False
        self.count[s, ns, a] += 1
        self._total_reward[s, ns, a] += r
        self._transition_count[s, ns, a] += 1
        if terminal:
            self.terminal[ns] = 1

    def _update(self):
        if not self._updated:
            for i in range(self.nS):
                for a in range(self.nA):
                    if np.sum(self._transition_count[i,:,a])!=0:
                        self.transitions[i,  :, a] =\
                            self._transition_count[i, :, a]/np.sum(self._transition_count[i, :, a])
                if not self.entropy_known:
                    for j in range(self.nS):
                        self.entropy[i, j, :] = -np.sum(self.transitions[i, j, :]*\
                                np.log(self.transitions[i, j, :]))
            self.reward = self._total_reward/ self.count
        self._updated = True

    def Qupdate(self):
        # Updating Q Values
        self._update()
        for i in range(self.it):
            m = np.max(self.Q, axis=1)
            for s in range(self.nS):
                for a in range(self.nA):
                    if self.terminal[s] != 1:
                        self.Q[s,a] = self.gamma * np.sum(self.transitions[s, : ,a] * m) +\
                            np.sum(self.transitions[s, : ,a] * (self.reward[s, : ,a] +\
                            (self.beta + 0*(1/((1/self.beta)+self.lambd*self.entropy[s, : , a])))\
                            /np.sqrt(self.count[s, : ,a])))
                    else: # that is sketchy but shouldn't make difference!
                        self.Q[s,a] = self.reward[s, s ,a] +\
                             (self.beta + 0* (1/((1/self.beta)+self.lambd*self.entropy[s, s , a])))\
                             /np.sqrt(self.count[s, s ,a])
        np.set_printoptions(precision=1, linewidth=100, suppress=True)

        #print(np.mean(1/(self.beta+self.lambd*self.entropy)/np.sqrt(self.count),axis=1).reshape(11,11))
       # print(np.sum(np.sum(self.reward, axis=1), axis=1).reshape(11,11))
        #print(np.max(self.Q, axis=1).reshape(11,11)*100)
        #print(np.sum(np.sum(self.reward + (self.beta + 1/((1/self.beta)+self.lambd*self.entropy)), axis =1), axis=1).reshape(11,11))
    def fill_entropy(self):
         det_ent = 0
         sto_ent = np.log(self.nA)

         for row in range(self.env.size):
             for col in range(self.env.size):
                 s = self.env.state[row, col]
                 if self.env.to_symbol[self.env.map[row, col]] in ['s', 'd', 'u', 'a', '#', 'g', 'x']:
                     self.entropy[s, :, :] = sto_ent
                 else:
                     self.entropy[s, :, :] = det_ent

