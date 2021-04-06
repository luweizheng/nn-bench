import torch.optim


class Optimizer(object):

    def __init__(self, args, params):
        super().__init__()
        self.args = args
        self.params = params

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        pass

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, '_optimizer'):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of torch.optim.Optimizer')
        return self._optimizer

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        raise NotImplementedError

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        # override learning rate, momentum, etc. with latest values
        for group in self.optimizer.param_groups:
            group.update(self.optimizer_config)

    def step(self, closure=None):
        """Performs a single optimization step."""
        return self.optimizer.step(closure)

    def zero_grad(self):
        if hasattr(self.optimizer, "accelerate") and self.optimizer.accelerate:
            return new_zero_grad(self.optimizer)
        else:
            """Clears the gradients of all optimized parameters."""
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    p.grad = None
            return self.optimizer.zero_grad()


def new_zero_grad(optimizer):
    stash = optimizer._amp_stash
    optimizer._amp_lazy_init()
    # Zero the model grads.
    stash.process_zero_grad = True
    for param in stash.all_fp16_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
    for param in stash.all_fp32_from_fp32_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
    # Clear the master grads that are independent of model grads
    for param in stash.all_fp32_from_fp16_params:
        if optimizer.accelerate and param.grad is not None:
            stash.combined_tensor_fp32_from_fp16.zero_()
            break
        else:
            param.grad = None