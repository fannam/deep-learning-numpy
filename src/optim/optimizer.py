class Optimizer:
    """
    Base class for all optimizers.
    """

    def __init__(self, parameters, lr):
        # TODO: store parameters and learning rate
        pass

    def step(self):
        # TODO: implement in subclasses
        raise NotImplementedError

    def zero_grad(self):
        # TODO: reset gradients of all managed parameters
        pass
