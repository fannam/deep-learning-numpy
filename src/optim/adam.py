from src.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        # TODO: store hyperparameters and initialize first/second moment buffers
        pass

    def step(self):
        # TODO: implement Adam update rule
        pass
