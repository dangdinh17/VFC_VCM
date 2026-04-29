class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.total += val * n
        self.count += n
        self.avg = self.total / self.count if self.count > 0 else 0
    def reset(self):
        self.val = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def value(self):
        return self.avg
    
    def sum(self):
        return self.total