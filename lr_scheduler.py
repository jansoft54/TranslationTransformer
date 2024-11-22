import torch
import math
class InverseSquareRootLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, init_lr, min_lr=1e-9, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.min_lr = min_lr
        super(InverseSquareRootLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            lr = self.init_lr * (step / self.warmup_steps)
        else:
            lr = self.init_lr * math.sqrt(self.warmup_steps / step)

        lr = max(lr, self.min_lr)

        return [lr for _ in self.base_lrs]

