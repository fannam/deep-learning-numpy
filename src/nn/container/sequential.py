from src.nn.module import Module


class Sequential(Module):
    """
    A container that chains modules together in order.
    """

    def __init__(self, *modules):
        # TODO: store modules in order and register each as a submodule
        pass

    def forward(self, x):
        # TODO: pass input through each module in sequence
        pass
