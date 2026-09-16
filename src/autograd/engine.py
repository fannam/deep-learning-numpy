class Node:
    """
    Represents a node in the computation graph.
    """

    def __init__(self, tensor, grad_fn=None, parents=None):
        # TODO: store reference to output tensor, grad_fn, and parent nodes
        pass

    def __repr__(self):
        # TODO: return readable representation of the node
        pass


def topological_sort(root):
    """
    Return graph nodes in topological order for the backward pass.
    """
    # TODO: implement topological sort (DFS post-order over grad_fn parents)
    pass


class Engine:
    """
    Executes the backward pass over the computation graph.
    """

    @staticmethod
    def backward(tensor, grad=None):
        # TODO: initialize gradient of output tensor (default to ones_like)
        # TODO: topologically sort the graph rooted at tensor
        # TODO: traverse nodes in reverse topological order, calling grad_fn.backward
        # TODO: accumulate resulting gradients into each leaf tensor's .grad
        pass
