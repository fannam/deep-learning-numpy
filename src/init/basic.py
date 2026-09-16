from src.init.initializer import Initializer


class Zeros(Initializer):
    def __call__(self, shape):
        # TODO: return array of zeros
        pass


class Ones(Initializer):
    def __call__(self, shape):
        # TODO: return array of ones
        pass


class Constant(Initializer):
    def __init__(self, value):
        # TODO: store constant value
        pass

    def __call__(self, shape):
        # TODO: return array filled with constant value
        pass
