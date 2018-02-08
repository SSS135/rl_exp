class RunningNormalization:
    def __init__(self, momentum, axis=None, ddof=1, eps=1e-2):
        self.momentum = momentum
        self.eps = eps
        self.axis = axis
        self.ddof = ddof
        self.running_mean = None
        self.running_std = None

    def update(self, arr):
        mean = arr.mean(axis=self.axis)
        std = arr.std(axis=self.axis, ddof=self.ddof)
        if self.running_mean is None:
            self.running_mean, self.running_std = mean, std
        else:
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_std = self.momentum * self.running_std + (1 - self.momentum) * std

    def normalize(self, arr):
        return (arr - self.running_mean) / (self.eps + self.running_std)