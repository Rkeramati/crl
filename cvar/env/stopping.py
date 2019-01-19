class stopping_env():
     def __init__(self):
         self.state = 0
         self.time = 20
         self.cost = 1

         self.fu = 1.5
         self.fd = 0.8
         self.ph = 0.1
         self.p = 0.65

         self.nS = 21
         self.nA = 2

         self.action_space = {0: 'accept', 1:'reject'}
     def act(self, action):
         if self.action_space[action] == 'accept' or self.state == self.time:
             self.state = 0
             return self.state, self.cost, True
         if self.action_space[action] == 'reject':
             self.state += 1
             if np.random.rand() < self.p:
                 self.cost = self.fu * self.cost
             else:
                 self.cost = self.fd * self.cost
             return self.state, self.ph, False
     def reset(self):
         self.state = 0
         self.cost = 1

