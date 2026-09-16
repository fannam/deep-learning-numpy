from src.optim.optimizer import Optimizer


class Adagrad(Optimizer):
    def __init__(self, parameters, lr=0.01, eps=1e-10):
        # TODO: store hyperparameters and initialize accumulated squared-gradient buffers
        pass

    def step(self):
        # TODO: implement Adagrad update rule
        pass
