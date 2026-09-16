from src.nn.module import Module


class ModuleDict(Module):
    """
    Holds submodules in a dict, without defining a forward pass.
    """

    def __init__(self, modules=None):
        # TODO: store modules and register each as a submodule
        pass

    def __getitem__(self, key):
        # TODO: return module for key
        pass

    def __setitem__(self, key, module):
        # TODO: register module under key
        pass

    def __len__(self):
        # TODO: return number of modules
        pass
