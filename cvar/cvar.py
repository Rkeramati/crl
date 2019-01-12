import numpy as np

class CVaROptimize():
    def __init__(self, config):
        self.lr = {lr1: 1e-5, lr2: 1e-4, lr3: 5*1e-4, lr4: 1e-3}

        self.policy = np.random.rand(config.nA, config.nS) #policy parameters
        self.value = np.random.rand(config.nS+1)

        self.lambd_max = 5
        self.lambd = 0.1
        self.nu = 0.1
        self.beta = 0.1

        self.gamma = config.gamma
        self.alpha = config.alpha
        self.nS = config.nS
        self.nA = config.nA
        self.initial_state = config.initial_state
        self.Cmax = config.Cmax

    def softmax(self, x): #softmax function
        return np.exp(x) / np.sum(np.exp(x), axis=0)


    def value_featurize(self, x, s): #featuring value function
        size = self.nS + 1
        feature = np.zeros(size)
        feature[:self.nS] = x
        feature[-1] = s
        return feature

    def map_back(self, lambd, policy, nu):
        lambd = np.clip(lambd, 0, self.lambd_max)
        nu = np.clip(nu, -self.Cmax/(1-self.gamma), self.Cmax/(1-self.gamma))
        return lambd, policy, nu

    def act(self, x, s):
        prob = self.softmax(np.dot(self.policy, self.value_featurize(x,s)))
        a = np.random.choice(np.arange(self.nA), p=prob)
        return a

    def observe(self, x ,s ,a , nx, cost, k, terminal):
        # state:x, cost so far: s, action: a, iteration number: k, cost = reward
        ns = (s - cost)/self.gamma # s of the new state

        # adjusting the cost for the terminal state
        if not terminal:
            cost = cost
        else:
            cost = self.lambd * np.positive(-s)/self.alpha

        # TD Error
        TDerror = cost + self.gamma * np.dot(self.value, self.value_featurize(nx,ns))-\
            np.dot(self.value, self.value_featurize(x,s))

        # Updating Critic
        value_update = self.lr['lr4'] * TDError * self.value_featurize(x, s)

        # Actor Update:
        value_der = (self.value_featurize(self.inital_state, self.nu + 0.001) -\
                     self.value_featurize(self.inital_state, self.nu - 0.001))/(0.002)
        nu_update = -self.lr['lr3']*(self.lambd + np.dot(self.value, value_der))

        # Policy Update
        current_policy = self.softmax(np.dot(self.policy, self.value_featurize(x,s)))
        grad_log_policy = np.zeros(self.nA, self.nS)
        grad_log_policy[:, s] = -current_policy[a]
        grad_log_policy[a, s] = 1-current_policy[a]

        policy_update = -self.lr['lr2']/(1-self.gamma) * grad_log_policy * TDerror

        # lambda update

        lambd_update= self.lr['lr1'] * (self.nu - self.beta)

        if terminal:
            lambd_update = self.lr['lr1'] *\
                    (self.nu - self.beta + self.lambda**k/(1-self.alpha)*np.positive(-s))

        self.lambd = self.lambd + lambd_update
        self.policy = self.policy + policy_update
        self.nu = self.nu + nu_update

        self.lambd, self.policy, self.nu = self.map_back(self.lambd, self.policy, self.nu)

        return ns



