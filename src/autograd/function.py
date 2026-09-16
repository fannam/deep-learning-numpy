import numpy as np
from abc import ABC, abstractmethod

class Function(ABC):
    def __init__(self):
        self.saved_tensors = ()

    @classmethod 
    def apply(cls, *inputs):
        """
        Apply the operations to input Tensors and build the graph
        """


        # 1. Build Function object
        function = cls()

        # 2. Forward
        output_data = function.forward(*inputs)

        # TODO:
        # 3. Determine if output requires grad


        # 4. Create output Tensor
        # 5. Attach grad_fn
        # save graph metadata

        return output_data

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

    @abstractmethod 
    def backward(self, grad_output):
        raise NotImplementedError

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors