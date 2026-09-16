class Dataset:
    """
    Abstract dataset representing a collection of samples.
    """

    def __len__(self):
        # TODO: return number of samples
        raise NotImplementedError

    def __getitem__(self, index):
        # TODO: return sample at index
        raise NotImplementedError
