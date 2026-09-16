class Sampler:
    """
    Base class for sampling strategies over a dataset.
    """

    def __iter__(self):
        # TODO: yield indices
        raise NotImplementedError

    def __len__(self):
        # TODO: return number of indices
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, dataset):
        # TODO: store dataset reference
        pass


class RandomSampler(Sampler):
    def __init__(self, dataset):
        # TODO: store dataset reference
        pass
