import numpy as np

class CVaROptimize():
    def __init__(self, config, name='stopping'):
        self.lr = {'lr1': 1e-5, 'lr2': 1e-4, 'lr3': 5*1e-4, 'lr4': 1e-3}
        config = config.set(name)
        self.num_feature = config.num_feature

        self.policy = np.random.rand(config.nA, self.num_feature)*config.initialization_std
        self.value = np.random.rand(self.num_feature)*config.initialization_std

        self.lambd_max = config.lambd_max
        self.lambd = config.initial_lambd #0.1
        self.nu = config.initial_nu #0.1
        self.beta = config.beta

        self.gamma = config.gamma
        self.alpha = config.alpha
        self.nS = config.nS
        self.nA = config.nA
        self.initial_state = config.initial_state
        self.Cmax = config.Cmax

        self.x_range = config.x_range
        self.s_range = config.s_range

        self.counter = 1
        self.lr_def = config.lr

    def softmax(self, x): #softmax function
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _update_lr(self):
        self.lr['lr1'] = self.lr_def * 1.0/self.counter
        self.lr['lr2'] = self.lr_def * 1.0/self.counter**(0.85)
        self.lr['lr3'] = self.lr_def * 0.5/self.counter**(0.7)
        self.lr['lr4'] = self.lr_def * 0.5/self.counter**(0.55)

        self.counter += 1
    def value_featurize(self, x, s): #featuring value function
        size =  self.num_feature - 1
        x_space = np.linspace(min(self.x_range), max(self.x_range), size)
        s_space = np.linspace(min(self.s_range), max(self.s_range), size)

        rdim1 = np.expand_dims(x - x_space, axis = 1);
        rdim2 = np.expand_dims(s - s_space, axis = 1);
        feature = np.concatenate([rdim1, rdim2], axis = 1)
        feature = np.linalg.norm(feature, 2, axis= 1)
        feature = np.exp(-feature/2)
        feature = np.concatenate([feature, np.array([1])], axis = 0)

        return feature

    def map_back(self, lambd, policy, nu):
        lambd = np.clip(lambd, 0, self.lambd_max)
        nu = np.clip(nu, -self.Cmax/(1-self.gamma), self.Cmax/(1-self.gamma))
        policy = np.clip(policy, -60, 60)
        return lambd, policy, nu

    def act(self, x, s):
        prob = self.softmax(np.dot(self.policy, self.value_featurize(x,s)))
        a = np.random.choice(np.arange(self.nA), p=prob)
        return a

    def observe(self, x ,s ,a , nx, cost, k, terminal):
        self._update_lr()
        # state:x, cost so far: s, action: a, iteration number: k, cost = reward
        ns = (s - cost)/self.gamma # s of the new state

        # adjusting the cost for the terminal state
        if not terminal:
            cost = cost
        else:
            cost = self.lambd * np.positive(-s)/(1-self.alpha)
        # TD Error
        TDerror = cost + self.gamma * np.dot(self.value, self.value_featurize(nx,ns))-\
            np.dot(self.value, self.value_featurize(x,s))

        # Updating Critic
        value_update = self.lr['lr4'] * TDerror * self.value_featurize(x, s)

        # Actor Update:
        value_der = (self.value_featurize(self.initial_state, self.nu + 0.001) -\
                     self.value_featurize(self.initial_state, self.nu - 0.001))/(0.002)
        #print(value_der)
        nu_update = -self.lr['lr3']*(self.lambd + np.dot(self.value, value_der))

        # Policy Update
        current_policy = self.softmax(np.dot(self.policy, self.value_featurize(x,s)))
        grad_log_policy = np.zeros((self.nA, self.num_feature))
        for actions in range(self.nA):
            grad_log_policy[actions, :] = -current_policy[actions] * self.value_featurize(x,s)
        grad_log_policy[a, :] = self.value_featurize(x,s)*(1-current_policy[a])
        #print(current_policy)
        policy_update = -self.lr['lr2']/(1-self.gamma) * grad_log_policy * TDerror

        # lambda update

        lambd_update= self.lr['lr1'] * (self.nu - self.beta)

        if terminal:
            lambd_update = self.lr['lr1'] *\
                    (self.nu - self.beta + 1.0/((1-self.alpha)*(1-self.gamma)) *np.positive(-s))

        self.lambd = self.lambd + lambd_update
        self.policy = self.policy + policy_update
        #print('Policy Update: %g, Lambd Update: %g, Nu update: %g'%(np.linalg.norm(policy_update),\
        #        np.linalg.norm(lambd_update), np.linalg.norm(nu_update)))
        self.nu = self.nu + nu_update
        #print(self.policy)
        self.value += value_update

        self.lambd, self.policy, self.nu = self.map_back(self.lambd, self.policy, self.nu)
        #print(self.lambd, self.nu)
        if terminal:
            ns = self.nu
        #print(self.value)
        return ns



