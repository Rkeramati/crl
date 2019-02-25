def mv(a, n=20) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class C51():
    def __init__(self, config, init='uniform', ifCVaR = False):
        self.ifCVaR = ifCVaR
        self.config = config

        if init == 'optimistic':
            self.p = np.zeros((self.config.nS, self.config.nA,\
                        self.config.nAtoms))
            self.p[:, :, -2] = 1
        elif init == 'uniform':
        	self.p = np.ones((self.config.nS, self.config.nA,\
                        self.config.nAtoms)) * 1.0/self.config.nAtoms
        else:
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
        self.p[x, a, :] = self.p[x, a, :] + alpha * (m - self.p[x, a, :]) ##### How ####
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

class Qlearning():
    def __init__(self, config):
        self.config = config
        self.Q = np.zeros((self.config.nS, self.config.nA))
    def observe(self, x, a, r, nx, alpha):
        self.Q[x, a] = self.Q[x, a] + alpha * \
        (r + self.config.gamma * np.max(self.Q[nx, :]) - self.Q[x, a])