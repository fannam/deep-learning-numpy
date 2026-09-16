from src.nn.module import Module


class ModuleList(Module):
    """
    Holds submodules in a list, without defining a forward pass.
    """

    def __init__(self, modules=None):
        # TODO: store modules and register each as a submodule
        pass

    def append(self, module):
        # TODO: add a module to the list
        pass

    def __getitem__(self, idx):
        # TODO: return module at index
        pass

    def __len__(self):
        # TODO: return number of modules
        pass
