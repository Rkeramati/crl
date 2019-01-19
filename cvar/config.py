class config():
    def __init__(self, nS, nA):
        self.alpha = 0.95
        self.beta = 2.5
        self.lambd_max = 1000
        self.Cmax = 4000
        self.nS = nS
        self.nA = nA
    def set(self, name):
        if name == 'stopping':
            self.num_feature = 20

            self.initial_state = 0
            self.gamma = 0.95

            self.xrange = (0, 20)
            self.srange = (-500, 0)

            self.initialization_std
