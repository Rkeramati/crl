import numpy as np

class MBIE():
    # Class for performing model based RL
    def __init__(self, env, const=0.2):

        #self.printInfo()
        self.gamma = 0.99
        self.beta = const
        self._updated = False
        self.it = 20 # iteration for solving

        self.nS = env.nS # number of states
        self.nA = env.nA # number of actions

        self.count = np.zeros((self.nS, self.nA)) # Counting s,a pairs
        self.reward = np.zeros((self.nS, self.nA)) # Reward Only function of states
        self.transitions = np.zeros((self.nS, self.nS, self.nA)) # s, s', a
        self.Q = np.zeros((self.nS, self.nA))

        self._total_reward = np.zeros((self.nS, self.nA))
        self._transition_count = np.zeros((self.nS, self.nS, self.nA))

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
                    if self.count[i,a] !=0:
                        self.reward[i, a] = self._total_reward[i, a]/ self.count[i, a]
        self._updated = True

    def Qupdate(self):
        # Updating Q Values with MBIE-EB
        self._update()
        for i in range(self.it):
            m = np.max(self.Q, axis=1)
            for s in range(self.nS):
                for a in range(self.nA):
                    self.Q[s,a] = self.reward[s,a] + \
                            self.gamma * np.sum(self.transitions[s, :, a] * m)+\
                            self.beta/np.sqrt(1+self.count[s,a])

class MBIE_NS():
    # Class for performing model based RL
    # reward being a function of s, a, ns
    def __init__(self, env, const=0.5):

        #self.printInfo()
        self.gamma = 0.99
        self.beta = const
        self._updated = False
        self.it = 20 # iteration for solving

        self.nS = env.nS # number of states
        self.nA = env.nA # number of actions

        self.count = np.zeros((self.nS, self.nS, self.nA)) + 1 # Counting s,a pairs
        self.reward = np.zeros((self.nS, self.nS, self.nA)) # Reward Only function of states
        self.transitions = np.zeros((self.nS, self.nS, self.nA)) # s, s', a
        self.Q = np.zeros((self.nS, self.nA)) + 1/(1-self.gamma)

        self._total_reward = np.zeros((self.nS, self.nS, self.nA))
        self._transition_count = np.zeros((self.nS, self.nS, self.nA)) + 1

    def observe(self, s, a, ns, r):
        # adding state action next state reward to the history
        # reward is associated with ns
        self._updated = False
        self.count[s, ns, a] += 1
        self._total_reward[s, ns, a] += r
        self._transition_count[s, ns, a] += 1

    def _update(self):
        if not self._updated:
            for i in range(self.nS):
                for a in range(self.nA):
                    if np.sum(self._transition_count[i,:,a])!=0:
                        self.transitions[i,  :, a] =\
                            self._transition_count[i, :, a]/np.sum(self._transition_count[i, :, a])
            self.reward = self._total_reward/ self.count
        self._updated = True

    def Qupdate(self):
        # Updating Q Values with MBIE-EB
        self._update()
        for i in range(self.it):
            self.Q = self.gamma * np.sum(self.transitions * np.expand_dims(np.max(self.Q, axis=1, keepdims=True), axis=-1), axis=1) +\
                    np.sum(self.transitions * (self.reward + self.beta/ np.sqrt(self.count)), axis = 1)

