import sys
from typing import Tuple, Optional

from torch.optim import lr_scheduler


__all__ = ['GradualLRWarmupScheduler']


class GradualLRWarmupScheduler(lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            nr_steps: int,
            target_lr_by_group: Optional[Tuple[float, ...]] = None,
            initial_lr: float = 1e-10):
        self.initial_lr = initial_lr
        self.target_lr_by_group = \
            tuple(param_group['lr'] for param_group in optimizer.param_groups) \
                if target_lr_by_group is None else target_lr_by_group
        self.nr_steps = nr_steps
        self.lr_step_increase_delta_by_group = tuple((lr - initial_lr) / nr_steps for lr in self.target_lr_by_group)
        assert all(delta > sys.float_info.epsilon for delta in self.lr_step_increase_delta_by_group)
        super(GradualLRWarmupScheduler, self).__init__(optimizer)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr

    def step(self):
        for param_group, target_lr, lr_step_increase_delta in \
                zip(self.optimizer.param_groups, self.target_lr_by_group, self.lr_step_increase_delta_by_group):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr, min(old_lr + lr_step_increase_delta, target_lr))
            if abs(new_lr - target_lr) < lr_step_increase_delta / 2:
                new_lr = target_lr
            param_group['lr'] = new_lr
