class config():
    def __init__(self, nS, nA, envName):
        #General
        self.nS = nS
        self.nA = nA
        self.gamma = 0.9
        self.numIteration = 500

        #Learning Rates:
        self.varTdLearningRate = 0.01
        self.varTdLearningNumFeature = None # one-hot encoding

        #PG:
        self.pgNumFeature = None # if none = nS
        self.valueLr = 0.01
        self.valueLrMin = 1e-3
        self.policyLr= 0.001
        self.policyLrMin = 1e-4
        self.varianceThresh = 5
        self.lambd = 100

        #Envinronment Specific:
        self.set(envName)
    def set(self, name):
        if name == "mrp":
            self.rewardScale = 100
            self.gamma = 0.8
