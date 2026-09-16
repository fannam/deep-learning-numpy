import numpy as np


class Tensor:
    """
    N-dimensional array with optional autograd tracking, backed by numpy.
    """

    DEFAULT_DTYPE = np.float32

    def __init__(self, data, requires_grad=False, dtype=None, grad=None, grad_fn=None):
        if isinstance(data, Tensor):
            data = data.data

        if dtype is None and not isinstance(data, np.ndarray):
            # let numpy infer dtype for raw ndarrays, but fall back to a
            # sane float default for python scalars/lists so `requires_grad`
            # tensors don't silently end up as int64.
            dtype = self.DEFAULT_DTYPE if requires_grad else None

        self.data = np.asarray(data, dtype=dtype)

        if requires_grad and not np.issubdtype(self.data.dtype, np.floating):
            raise TypeError(
                f"only floating point tensors can require grad, got dtype={self.data.dtype}"
            )

        self.requires_grad = requires_grad
        self.grad = grad
        self.grad_fn = grad_fn

    # ------------------------------------------------------------------
    # basic properties
    # ------------------------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        return self.transpose()

    @property
    def is_leaf(self):
        return self.grad_fn is None

    # ------------------------------------------------------------------
    # conversions
    # ------------------------------------------------------------------

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    @staticmethod
    def _ensure_tensor(value):
        return value if isinstance(value, Tensor) else Tensor(value)

    # ------------------------------------------------------------------
    # autograd
    # ------------------------------------------------------------------

    def backward(self, grad=None):
        # TODO: call Engine.backward(self, grad) once the engine is implemented
        from src.autograd.engine import Engine
        Engine.backward(self, grad)

    def zero_grad(self):
        self.grad = None

    # ------------------------------------------------------------------
    # arithmetic (dispatch to autograd-aware ops)
    # ------------------------------------------------------------------

    def __add__(self, other):
        from src.ops.arithmetic import Add
        return Add.apply(self, self._ensure_tensor(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from src.ops.arithmetic import Sub
        return Sub.apply(self, self._ensure_tensor(other))

    def __rsub__(self, other):
        from src.ops.arithmetic import Sub
        return Sub.apply(self._ensure_tensor(other), self)

    def __mul__(self, other):
        from src.ops.arithmetic import Mul
        return Mul.apply(self, self._ensure_tensor(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from src.ops.arithmetic import Div
        return Div.apply(self, self._ensure_tensor(other))

    def __rtruediv__(self, other):
        from src.ops.arithmetic import Div
        return Div.apply(self._ensure_tensor(other), self)

    def __neg__(self):
        from src.ops.arithmetic import Neg
        return Neg.apply(self)

    def __pow__(self, exponent):
        from src.ops.arithmetic import Pow
        return Pow.apply(self, exponent)

    def __matmul__(self, other):
        from src.ops.linalg import MatMul
        return MatMul.apply(self, self._ensure_tensor(other))

    # ------------------------------------------------------------------
    # reductions / shape ops
    # ------------------------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        from src.ops.basic import Sum
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        from src.ops.basic import Mean
        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape):
        from src.ops.shape import Reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Reshape.apply(self, shape)

    def transpose(self, *axes):
        from src.ops.shape import Transpose
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        return Transpose.apply(self, axes or None)

    # ------------------------------------------------------------------
    # python protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # TODO: wrap in an autograd-aware indexing op instead of raw numpy indexing
        return Tensor(self.data[index])

    def __repr__(self):
        grad_fn_repr = f", grad_fn=<{type(self.grad_fn).__name__}>" if self.grad_fn else ""
        requires_grad_repr = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{requires_grad_repr}{grad_fn_repr})"
