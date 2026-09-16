from src.autograd.function import Function


class Add(Function):
    def forward(self, a, b):
        # TODO: implement element-wise addition
        pass

    def backward(self, grad_output):
        # TODO: compute gradient w.r.t. a and b (handle broadcasting)
        pass


class Sub(Function):
    def forward(self, a, b):
        # TODO: implement element-wise subtraction
        pass

    def backward(self, grad_output):
        # TODO: compute gradient w.r.t. a and b (handle broadcasting)
        pass


class Mul(Function):
    def forward(self, a, b):
        # TODO: implement element-wise multiplication
        pass

    def backward(self, grad_output):
        # TODO: compute gradient w.r.t. a and b using saved inputs
        pass


class Div(Function):
    def forward(self, a, b):
        # TODO: implement element-wise division
        pass

    def backward(self, grad_output):
        # TODO: compute gradient w.r.t. a and b
        pass


class Neg(Function):
    def forward(self, a):
        # TODO: implement negation
        pass

    def backward(self, grad_output):
        # TODO: compute gradient w.r.t. a
        pass


class Pow(Function):
    def forward(self, a, exponent):
        # TODO: implement element-wise power
        pass

    def backward(self, grad_output):
        # TODO: compute gradient w.r.t. a
        pass
