import numpy as np
# A class for comuting the variance with TD
class TD_learner():
    def __init__(self, nS, gamma):
        self.nS = nS
        self.gamma = gamma
        self.value = np.zeros(nS)
        self.second_moment = np.zeros(nS)
        self.alpha = 0.01
    def compute_variance(self):
        self.variance = self.second_moment - self.value**2

    def observe(self, s, r, ns):
        # Updating value
        self.value[s] = self.value[s] +\
            self.alpha* (r + self.gamma*self.value[ns]-\
                        self.value[s])
        #Updating Second Moment
        self.second_moment[s] = self.second_moment[s] +\
            self.alpha* (r**2 + 2*r*self.gamma*self.value[ns]+\
                         self.gamma**2*self.second_moment[ns]-
                        self.second_moment[s])
        self.compute_variance()
