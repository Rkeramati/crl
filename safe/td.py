import numpy as np
# A class for comuting the variance with TD
class TDVar():
    def __init__(self, config):
        self.nS = config.nS
        self.gamma = config.gamma

        self.valueWeight = np.zeros(self.nS)
        self.secondMomentWeight = np.zeros(self.nS)

        self.alpha = config.varTdLearningRate
        if config.varTdLearningNumFeature is not None:
            self.numFeature = config.varTdLearningNumFeature
        else:
            self.numFeature = self.nS
        self.counter = 0

    def _featurize(self, s):
        # one-hot encoding
        feature = np.zeros(self.numFeature)
        feature[s] = 1
        return feature
    def _update_lr(self):
        self.alpha = self.alpha # for now no change

    def value(self, s):
        return np.dot(self._featurize(s), self.valueWeight)

    def secondMoment(self, s):
        return np.dot(self._featurize(s), self.secondMomentWeight)

    def variance(self, s):
        var = np.load("./result/variance_all_best_0.npy")
        return var[s, -1]
        return self.secondMoment(s) - self.value(s)**2

    def update(self, value, secondMoment):
        self._update_lr()
        self.valueWeight += self.alpha * value
        self.secondMomentWeight += self.alpha * secondMoment

    def observe(self, episode):
        #input: list of tuples (s, a, r, ns, terminal)
        valueUpdate = np.zeros_like(self.valueWeight)
        secondMomentUpdate = np.zeros_like(self.secondMomentWeight)

        for s, a, r, ns, terminal in episode:
            # Updating value
            delta = r + self.value(ns) - self.value(s)
            valueUpdate += self._featurize(s) * delta
            # Second Moment:
            delta = r**2 + 2*r*self.value(ns)+\
                         self.secondMoment(ns)- self.secondMoment(s)
            secondMomentUpdate += delta * self._featurize(s)

        self.update(valueUpdate, secondMomentUpdate)
