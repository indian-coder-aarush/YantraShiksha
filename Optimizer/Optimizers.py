import Ganit

class GradientDescent:

    def __init__(self,learning_rate,params):
        self.learning_rate = learning_rate
        self.params = params

    def step(self):
        for param_key in self.params:
            self.params[param_key] -= self.learning_rate * self.params[param_key].grad