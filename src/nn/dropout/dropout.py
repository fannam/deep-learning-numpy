from src.nn.module import Module


class Dropout(Module):
    def __init__(self, p=0.5):
        # TODO: store dropout probability
        pass

    def forward(self, x):
        # TODO: randomly zero elements during training and scale by 1/(1-p)
        pass
