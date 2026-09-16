from src.autograd.function import Function


class Sum(Function):
    def forward(self, a, axis=None, keepdims=False):
        # TODO: implement sum reduction
        pass

    def backward(self, grad_output):
        # TODO: broadcast gradient back to input shape
        pass


class Mean(Function):
    def forward(self, a, axis=None, keepdims=False):
        # TODO: implement mean reduction
        pass

    def backward(self, grad_output):
        # TODO: broadcast and scale gradient back to input shape
        pass


class Exp(Function):
    def forward(self, a):
        # TODO: implement element-wise exponential
        pass

    def backward(self, grad_output):
        # TODO: compute gradient using saved output
        pass


class Log(Function):
    def forward(self, a):
        # TODO: implement element-wise natural log
        pass

    def backward(self, grad_output):
        # TODO: compute gradient w.r.t. a
        pass


class Sqrt(Function):
    def forward(self, a):
        # TODO: implement element-wise square root
        pass

    def backward(self, grad_output):
        # TODO: compute gradient w.r.t. a using saved output
        pass


class Clone(Function):
    def forward(self, a):
        # TODO: implement copy of tensor data
        pass

    def backward(self, grad_output):
        # TODO: pass gradient through unchanged
        pass
