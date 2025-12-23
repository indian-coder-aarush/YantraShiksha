class GradientDescentOptimizer:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.params = []

    def step(self, grad):
        for param in self.params:
            param.grad -= self.learning_rate * param.grad