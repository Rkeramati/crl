class config():
    def __init__(self, nS, nA):
        self.alpha = 0.9
        self.beta = 2.5
        self.lambd_max = 1000
        self.Cmax = 4000
        self.nS = nS
        self.nA = nA

        self.initial_state = 0
        self.gamma = 0.95

