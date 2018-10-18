import numpy as np

class MRL():
    # Class for performing model based RL
    def __init__(self, nS, nA, entropy=False):
        self.entropy_use = entropy

        #self.printInfo()
        self.gamma = 0.99
        self.beta = 0.5
        self.Hbeta = 2-np.log(nS)
        self._updated = False
        self.it = 20

        self.nS = nS # number of states
        self.nA = nA # number of actions

        self.count = np.zeros((self.nS, self.nA)) + 1 # Counting s,a pairs
        self.reward = np.zeros((self.nS, self.nA)) # Reward Only function of states
        self.transitions = np.zeros((self.nS, self.nS, self.nA)) # s, s', a
        self.Q = np.zeros((self.nS, self.nA)) + 1/(1-self.gamma)

        self.entropy = np.zeros((self.nS, self.nA)) # Entropy s, a = \sum_{s'} H(s'|s,a)

        self._total_reward = np.zeros((self.nS, self.nA))
        self._transition_count = np.zeros((self.nS, self.nS, self.nA)) + 1


    def printInfo(self):
        print('[*] Info: Reward is only a function of states')
        print('[*] Info: Transitions are intialized by 1/nS, and ')

    def observe(self, s, a, ns, r):
        # adding state action next state reward to the history
        # reward is associated with ns
        self._updated = False
        self.count[s, a] += 1
        self._total_reward[s, a] += r
        self._transition_count[s, ns, a] += 1

    def _update(self):
        if not self._updated:
            for i in range(self.nS):
                for a in range(self.nA):
                    if np.sum(self._transition_count[i,:,a])!=0:
                        self.transitions[i,  :, a] =\
                            self._transition_count[i, :, a]/np.sum(self._transition_count[i, :, a])
                    self.reward[i, a] = self._total_reward[i, a]/ self.count[i, a]
                    self.entropy[i, a] = -np.sum(self.transitions[i, :, a] \
                            * np.log(self.transitions[i, :, a]))
        self._updated = True

    def Qupdate(self):
        if self.entropy_use:
            self._MBIE_H()
        else:
            self._MBIE_EB()
    def _MBIE_EB(self):
        # Updating Q Values with MBIE-EB
        self._update()
        for i in range(self.it):
            self.Q = self.reward + \
                    self.gamma * np.sum(self.transitions * np.expand_dims(np.max(self.Q, axis=1, keepdims=True), axis=-1), axis=1) +\
                    self.beta/ np.sqrt(self.count)
    def _MBIE_H(self):
        self._update()
        for i in range(self.it):
            self.Q = self.reward + \
                    self.gamma * np.sum(self.transitions * np.expand_dims(np.max(self.Q, axis=1, keepdims=True), axis=-1), axis=1) +\
                    (1/(self.Hbeta + self.entropy))/np.sqrt(self.count)



