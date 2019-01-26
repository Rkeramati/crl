import numpy as np

class pg():
    def __init__(self, config):
        self.value = 0
        self.variance = 0

        self.alpha = config.valueLr
        self.beta = config.policyLr
        self.b = config.varianceThresh
        self.lambd = config.lambd

        self.nS = config.nS
        self.nA = config.nA

        if config.pgNumFeature is None:
            self.numFeature = self.nS
        else:
            self.numFearure = config.pgNumFeature

        self.policyWeight = np.random.randn(self.nA, self.numFeature) * 0.05

    def _updateLr(self):
        self.alpha = self.alpha
        self.beta = self.beta

    def _featurize(self, s):
        feature = np.zeros(self.numFeature)
        feature[s] = 1
        return feature

    def _computeReturn(self, episode):
        ret = 0
        for s, a, r, ns, terminal in episode:
            ret += r
        return ret

    def _computeMleRation(self, episode):
        ratio = np.zeros_like(self.policyWeight)
        for s, a, r, ns, terminal in episode:
            ratio[:, :] += -self.policy(s)[a] * self._featurize(s)
            ratio[a, :] += self._featurize(s)
        return ratio

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference

    def policy(self, s):
        return self.softmax(np.dot(self.policyWeight, self._featurize(s)))

    def observe(self, episode):
        totalReturn = self._computeReturn(episode)
        mleRation = self._computeMleRation(episode)

        self._updateLr()
        # value
        newValue = self.value + self.alpha*(totalReturn - self.value)

        # variance
        newVariance = self.variance + self.alpha*(totalReturn**2 - self.value**2 - self.variance)

        # Policy
        derivative = 0
        if self.variance - self.b > 0:
            derivative = 2*(self.variance - self.b)
        newPolicyWeight = self.policyWeight + self.beta*(totalReturn - self.lambd*derivative *\
                (totalReturn**2-2*self.value**2)) * mleRation

        self.value = newValue
        self.variance = newVariance
        self.policyWeight = newPolicyWeight

