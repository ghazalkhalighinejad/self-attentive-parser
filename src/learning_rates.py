from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim.lr_scheduler as lr_scheduler


class WarmupThenReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, warmup_steps, *args, **kwargs):
        """
        Args:
            optimizer (Optimizer): Optimizer to wrap
            warmup_steps: number of steps before reaching base learning rate
            *args: Arguments for ReduceLROnPlateau
            **kwargs: Arguments for ReduceLROnPlateau
        """
        super().__init__(optimizer, *args, **kwargs)
        self.warmup_steps = warmup_steps
        self.steps_taken = 0
        self.base_lrs = list(map(lambda group: group["lr"], optimizer.param_groups))
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

    def get_lr(self):
        assert self.steps_taken <= self.warmup_steps
        return [
            base_lr * (self.steps_taken / self.warmup_steps)
            for base_lr in self.base_lrs
        ]

    def step(self, metrics=None):
        self.steps_taken += 1
        if self.steps_taken <= self.warmup_steps:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr
        elif metrics is not None:
            super().step(metrics)

def linear_then_plateau(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            1e-6, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)