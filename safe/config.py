class config():
    def __init__(self, nS, nA, envName):
        #General
        self.nS = nS
        self.nA = nA
        self.gamma = 0.5
        self.numIteration = 5000

        #Learning Rates:
        self.varTdLearningRate = 0.01
        self.varTdLearningNumFeature = None # one-hot encoding

        #PG:
        self.pgNumFeature = None # if none = nS
        self.valueLr = 0.1
        self.policyLr= 0.01
        self.varianceThresh = 0.01
        self.lambd = 2

        #Envinronment Specific:
        self.set(envName)
    def set(self, name):
        pass
