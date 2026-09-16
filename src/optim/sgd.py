from src.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        # TODO: store hyperparameters and initialize momentum buffers
        pass

    def step(self):
        # TODO: implement SGD update rule (with optional momentum/weight decay)
        pass
