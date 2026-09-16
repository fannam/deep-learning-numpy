class Module:
    """
    Base class for all neural network modules.
    """

    def __init__(self):
        # TODO: initialize storage for parameters, submodules, and training mode flag
        pass

    def forward(self, *inputs, **kwargs):
        # TODO: implement in subclasses
        raise NotImplementedError

    def __call__(self, *inputs, **kwargs):
        # TODO: call forward
        pass

    def parameters(self):
        # TODO: yield all parameters recursively (own + submodules)
        pass

    def named_parameters(self, prefix=""):
        # TODO: yield (name, parameter) pairs recursively
        pass

    def zero_grad(self):
        # TODO: reset gradients of all parameters to None/zero
        pass

    def train(self, mode=True):
        # TODO: set training mode recursively on self and submodules
        pass

    def eval(self):
        # TODO: set evaluation mode recursively
        pass

    def add_module(self, name, module):
        # TODO: register a submodule under the given name
        pass

    def __setattr__(self, name, value):
        # TODO: auto-register Parameter and Module instances on assignment
        pass

    def __repr__(self):
        # TODO: return readable representation of the module tree
        pass
