from src.tensor.tensor import Tensor


class Parameter(Tensor):
    """
    A Tensor that is automatically registered as a learnable parameter
    when assigned as an attribute of a Module.
    """

    def __init__(self, data, requires_grad=True):
        # TODO: initialize underlying Tensor with requires_grad=True
        pass
