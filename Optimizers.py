import Math
import numpy as np

class GradientDescent:

    def __init__(self,learning_rate,params):
        self.learning_rate = learning_rate
        self.params = params

    def step(self):
        for param_key in self.params:
            if self.params[param_key].grad is not None:
                self.params[param_key] -= Math.Tensor(np.full(self.params[param_key].shape(),self.learning_rate)) * self.params[param_key].grad