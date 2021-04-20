from collections import defaultdict, OrderedDict
import contextlib
from itertools import chain

import torch
import torch.nn.functional as F
# from optim.cpex import amp
from torch.nn.parallel import DistributedDataParallel as DDP


from utils import distributed_utils, utils
from optim import lr_scheduler
import optim
from utils.meters import TimeMeter
from utils.criterions import CRITERION_REGISTRY

# import dllogger as DLLogger
import math
import time
import numpy as np


class DDPTrainer(object):
    """Main class for data parallel training.

    This class supports data parallel training, where multiple workers each
    have a full model replica and gradients are accumulated synchronously via
    torch.distributed.all_reduce.
    """

    def __init__(self, args, model):

        self.args = args
        self.model = model.to(args.device)
        self.criterion = CRITERION_REGISTRY[args.criterion](self.args).to(args.device)
        self.optimizer = optim.build_optimizer(self.args, self.model.parameters())
        self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)
        if args.platform == "npu":
            from optim.cpex import amp
            if args.amp:
                model, optimizer = amp.initialize(
                    self.model,
                    self.optimizer._optimizer,
                    opt_level=self.args.amp_level if self.args.amp_level else 'O2',
                    # max_loss_scale=2**15,
                    # max_loss_scale=1024,
                    # min_loss_scale = 0.0001,
                    loss_scale=8,
                    cast_model_outputs=torch.float16,
                    combine_grad=True,
                    verbosity=0
                )
        elif args.platform == "gpu":
            from apex import amp
            if args.amp:
                model, optimizer = amp.initialize(
                    self.model,
                    self.optimizer._optimizer,
                    opt_level=self.args.amp_level if self.args.amp_level else 'O2',
                    # max_loss_scale=2**15,
                    # max_loss_scale=1024,
                    # min_loss_scale = 0.0001,
                    loss_scale=8,
                    cast_model_outputs=torch.float16,
                    verbosity=0
                )

        if self.args.distributed_world_size > 1:
            self.model = DDP(model, device_ids=[self.args.device_id], broadcast_buffers=False)

        self._buffered_stats = defaultdict(lambda: [])
        self._flat_grads = None
        self._num_updates = 0
        self._num_val_iterations = 0
        self._optim_history = None
        self.throughput_meter = TimeMeter()

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if self.args.amp:
            extra_state['amp_state_dict'] = amp.state_dict()
            extra_state['amp_master_params'] = list(amp.master_params(self.optimizer.optimizer))
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            utils.save_state(
                filename, self.args, self.get_model(), self.criterion, self.optimizer,
                self.lr_scheduler, self._num_updates, self._optim_history, extra_state,
            )

    def load_checkpoint(self, filename, load_optim=True):
        """Load all training state from a checkpoint file."""
        extra_state, optim_history, last_optim_state = \
            utils.load_model_state(filename, self.get_model())

        if last_optim_state is not None:
            # rebuild optimizer after loading model, since params may have changed
            # self.optimizer = optim.build_optimizer(self.args, self.model.parameters())
            self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)

            if load_optim:
                self._optim_history = optim_history
                # only reload optimizer and lr_scheduler if they match
                last_optim = self._optim_history[-1]
                if last_optim['criterion_name'] == self.criterion.__class__.__name__:
                    self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
                    if last_optim['optimizer_name'] == self.optimizer.__class__.__name__:
                        self.optimizer.load_state_dict(last_optim_state)

                self._num_updates = last_optim['num_updates']

        if self.args.amp and extra_state is not None and 'amp_state_dict' in extra_state:
            self.optimizer.optimizer._lazy_init_maybe_master_weights()
            self.optimizer.optimizer._amp_stash.lazy_init_called = True
            self.optimizer.optimizer.load_state_dict(last_optim_state)
            for param, saved_param in zip(amp.master_params(self.optimizer.optimizer),
                                          extra_state['amp_master_params']):
                param.data.copy_(saved_param.data)

            amp.load_state_dict(extra_state['amp_state_dict'])

        return extra_state

    def train_step(self, sample, update_params=True, last_step=False):
        """Do forward, backward and parameter update."""
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints

        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)

        self.model.train()
        # forward and backward pass
        # move data to deivce
        sample, sample_size = self._prepare_sample(sample)

        loss, oom_fwd = self._forward(sample)


        # If this is a last batch forward pass is skipped on some workers
        # Batch with sample_size 0 is not accounted for in weighted loss
        logging_output = {
            'ntokens': sample['ntokens'] if sample is not None else 0,
            'nsentences': sample['target'].size(0) if sample is not None else 0,
            'loss': utils.item(loss.data) if loss is not None else 0,
            'sample_size': sample_size

        }

        oom_bwd = self._backward(loss)

        # buffer stats and logging outputs
        self._buffered_stats['sample_sizes'].append(sample_size)
        self._buffered_stats['logging_outputs'].append(logging_output)
        self._buffered_stats['ooms_fwd'].append(oom_fwd)
        self._buffered_stats['ooms_bwd'].append(oom_bwd)

        # update parameters
        if update_params and not last_step:
            # gather logging outputs from all replicas
            sample_sizes = self._buffered_stats['sample_sizes']
            logging_outputs = self._buffered_stats['logging_outputs']
            ooms_fwd = self._buffered_stats['ooms_fwd']
            ooms_bwd = self._buffered_stats['ooms_bwd']

            # if self.args.distributed_world_size > 1:
            #     sample_sizes, logging_outputs, ooms_fwd, ooms_bwd = map(
            #         lambda l: list(chain.from_iterable(l)),
            #         zip(*distributed_utils.all_gather_list(
            #             (sample_sizes, logging_outputs, ooms_fwd, ooms_bwd)
            #         ))
            #     )
            ooms_fwd = sum(ooms_fwd)
            ooms_bwd = sum(ooms_bwd)
            ooms = ooms_fwd + ooms_bwd  # this is always <= distributed_world_size


            # aggregate stats and logging outputs

            grad_denom = sum(sample_sizes)
            # print(grad_denom)
            if self.args.amp and self.optimizer.optimizer.accelerate:
                stash = self.optimizer.optimizer._amp_stash
                if stash.combined_tensor_fp16 is not None:
                    stash.combined_tensor_fp16.div_(grad_denom)
            else:
                for p in self.model.parameters():
                    if p.requires_grad and not p.grad is None:
                        p.grad /= grad_denom

            self._opt()

            # Handle logging
            sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
            ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
            self.throughput_meter.update(ntokens)
            info_log_data = {
                'tokens/s': self.throughput_meter.avg,
                'tokens': ntokens,
                'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2)
            }
            debug_log_data = {
                'batch_size': sum(log.get('nsentences', 0) for log in logging_outputs),
                'lr': self.get_lr(),
                'grad_denom': grad_denom,
                'updates': 1
            }

            # DLLogger.log(step=self._num_updates, data=info_log_data, verbosity=0)
            # DLLogger.log(step=self._num_updates, data=debug_log_data, verbosity=1)

            self.clear_buffered_stats()
            return sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2)
        return loss / sample_size / math.log(2)

    def _forward(self, sample):
        loss = None
        oom = 0
        try:
            if sample is not None:
                
                # calculate loss and sample size
                # src_tokens, src_lengths, prev_output_tokens
                # npu only accept int32 tensors
                if self.args.platform == "npu":
                    sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].to(torch.int32)
                    sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].to(torch.int32)
                    sample['target'] = sample['target'].to(torch.int32)
                elif self.args.platform == "gpu":
                    sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].to(torch.int64)
                    sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].to(torch.int64)
                    sample['target'] = sample['target'].to(torch.int64)

                logits, _ = self.model(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], sample['net_input']['prev_output_tokens'])
                target = sample['target']
                
                probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                loss = self.criterion(probs, target)
        except RuntimeError as e:
            if not eval and 'out of memory' in str(e):
                print('| WARNING: ran out of memory in worker {}, skipping batch'.format(self.args.distributed_rank),
                      force=True)
                oom = 1
                loss = None
            else:
                raise e
        return loss, oom

    def _backward(self, loss):
        oom = 0
        if loss is not None:
            try:
                if self.args.amp:
                    if self.args.platform == "npu":
                        from optim.cpex import amp
                    elif self.args.platform == "gpu":
                        from apex import amp
                        with amp.scale_loss(loss, self.optimizer._optimizer) as scaled_loss:
                            scaled_loss.backward()
                else:
                    loss.backward()


            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(
                        '| WARNING: ran out of memory in worker {}, skipping batch'.format(self.args.distributed_rank),
                        force=True)
                    oom = 1
                    self.zero_grad()
                else:
                    raise e
        return oom

    def _opt(self):
        # take an optimization step
        self.optimizer.step()
        self.zero_grad()
        self._num_updates += 1

        # update learning rate
        self.lr_scheduler.step_update(self._num_updates)

    def valid_step(self, sample):
        """Do forward pass in evaluation mode."""
        self.model.eval()
        self._num_val_iterations += 1
        # forward pass
        sample, sample_size = self._prepare_sample(sample)
        with torch.no_grad():
            loss, oom_fwd = self._forward(sample)
        logging_output = {
            'ntokens': sample['ntokens'] if sample is not None else 0,
            'nsentences': sample['target'].size(0) if sample is not None else 0,
            'sample_size': sample_size
        }
        loss = loss.item() if loss is not None else 0
        assert not oom_fwd, 'Ran out of memory during validation'

        losses = [loss]
        sample_sizes = [sample_size]
        logging_outputs = [logging_output]

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        weight = sum(log.get('sample_size', 0) for log in logging_outputs)
        scaled_loss = sum(losses) / weight / math.log(2)

        return scaled_loss

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, update_params=False)
        self.zero_grad()
        self.clear_buffered_stats()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clear_buffered_stats(self):
        self._buffered_stats.clear()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        return self.lr_scheduler.step(epoch, val_loss)

    def lr_step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(num_updates)

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_throughput_meter(self):
        """Get the throughput meter"""
        return self.throughput_meter

    def get_model(self):
        """Get the model replica."""
        # return self.model
        return self.model.module if isinstance(self.model, DDP) else self.model

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None, 0
        return utils.move_to_device(self.args, sample), sample['ntokens']
