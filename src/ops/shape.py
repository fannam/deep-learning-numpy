from src.autograd.function import Function


class Reshape(Function):
    def forward(self, a, shape):
        # TODO: implement reshape
        pass

    def backward(self, grad_output):
        # TODO: reshape gradient back to original input shape
        pass


class Transpose(Function):
    def forward(self, a, axes=None):
        # TODO: implement transpose/permute
        pass

    def backward(self, grad_output):
        # TODO: apply inverse permutation to gradient
        pass


class Squeeze(Function):
    def forward(self, a, axis=None):
        # TODO: implement squeeze
        pass

    def backward(self, grad_output):
        # TODO: reshape gradient back to original input shape
        pass


class Unsqueeze(Function):
    def forward(self, a, axis):
        # TODO: implement unsqueeze
        pass

    def backward(self, grad_output):
        # TODO: reshape gradient back to original input shape
        pass


class Concat(Function):
    def forward(self, tensors, axis=0):
        # TODO: implement concatenation along axis
        pass

    def backward(self, grad_output):
        # TODO: split gradient back to each input tensor
        pass


class Stack(Function):
    def forward(self, tensors, axis=0):
        # TODO: implement stacking along a new axis
        pass

    def backward(self, grad_output):
        # TODO: split gradient back to each input tensor
        pass
