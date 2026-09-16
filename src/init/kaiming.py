from src.init.initializer import Initializer


class KaimingUniform(Initializer):
    def __init__(self, mode="fan_in", nonlinearity="relu"):
        # TODO: store mode and nonlinearity
        pass

    def __call__(self, shape):
        # TODO: implement Kaiming/He uniform initialization
        pass


class KaimingNormal(Initializer):
    def __init__(self, mode="fan_in", nonlinearity="relu"):
        # TODO: store mode and nonlinearity
        pass

    def __call__(self, shape):
        # TODO: implement Kaiming/He normal initialization
        pass
