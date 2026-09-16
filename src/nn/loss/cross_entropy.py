from src.nn.module import Module


class CrossEntropyLoss(Module):
    def __init__(self, reduction="mean"):
        # TODO: store reduction mode
        pass

    def forward(self, logits, target):
        # TODO: implement (log-)softmax + negative log likelihood loss
        pass
