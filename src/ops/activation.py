from src.autograd.function import Function


class ReLU(Function):
    def forward(self, a):
        # TODO: implement ReLU forward
        pass

    def backward(self, grad_output):
        # TODO: implement ReLU backward
        pass


class Sigmoid(Function):
    def forward(self, a):
        # TODO: implement sigmoid forward
        pass

    def backward(self, grad_output):
        # TODO: implement sigmoid backward using saved output
        pass


class Tanh(Function):
    def forward(self, a):
        # TODO: implement tanh forward
        pass

    def backward(self, grad_output):
        # TODO: implement tanh backward using saved output
        pass


class Softmax(Function):
    def forward(self, a, axis=-1):
        # TODO: implement numerically stable softmax
        pass

    def backward(self, grad_output):
        # TODO: implement softmax backward (Jacobian-vector product)
        pass


class GELU(Function):
    def forward(self, a):
        # TODO: implement GELU forward
        pass

    def backward(self, grad_output):
        # TODO: implement GELU backward
        pass
