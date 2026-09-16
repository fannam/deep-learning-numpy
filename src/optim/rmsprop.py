from src.optim.optimizer import Optimizer


class RMSprop(Optimizer):
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
        # TODO: store hyperparameters and initialize squared-gradient average buffers
        pass

    def step(self):
        # TODO: implement RMSprop update rule
        pass
