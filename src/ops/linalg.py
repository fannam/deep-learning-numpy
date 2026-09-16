from src.autograd.function import Function


class MatMul(Function):
    def forward(self, a, b):
        # TODO: implement matrix multiplication
        pass

    def backward(self, grad_output):
        # TODO: compute gradients w.r.t. a and b
        pass


class Dot(Function):
    def forward(self, a, b):
        # TODO: implement dot product
        pass

    def backward(self, grad_output):
        # TODO: compute gradients w.r.t. a and b
        pass


class Inverse(Function):
    def forward(self, a):
        # TODO: implement matrix inverse
        pass

    def backward(self, grad_output):
        # TODO: compute gradient of the inverse operation
        pass


class Norm(Function):
    def forward(self, a, ord=None, axis=None):
        # TODO: implement vector/matrix norm
        pass

    def backward(self, grad_output):
        # TODO: compute gradient of the norm
        pass
