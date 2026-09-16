from abc import ABC, abstractmethod


class Initializer(ABC):
    """
    Base class for parameter initializers.
    """

    @abstractmethod
    def __call__(self, shape):
        # TODO: return initialized numpy array of the given shape
        raise NotImplementedError
