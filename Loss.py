import Math

class MSE:

    def __call__(self,y_pred,y_true):
        return (y_pred - y_true)**2