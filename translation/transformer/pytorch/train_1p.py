import torch
import time
from apex import amp
import os
import sys
import math
from utils import options, utils, criterions
from utils.ddp_trainer import DDPTrainer
from utils.meters import StopwatchMeter, TimeMeter
import data
from data import data_utils, load_dataset_splits
from models import build_model
import numpy as np
from torch.utils.tensorboard import SummaryWriter

MAX = 2147483647
def _gen_seeds(shape):
    return np.random.uniform(1, MAX, size=shape).astype(np.float32)
seed_shape = (32 * 1024 * 12, )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def main(args):
    print(args)

    if args.platform == "gpu":
        device = torch.device('cuda:' + args.device_id)
        device_func = torch.cuda
    elif args.platform == "npu":
        device = torch.device('npu:' + args.device_id)
        device_func = torch.npu

        # import any Python files in the optim/ directory
        # print(f"{os.path.dirname(__file__)}")
        
        # for file in os.listdir(os.path.dirname(__file__) + "/optim"):
        #     if file.endswith('.py') and not file.startswith('_'):
        #         module = file[:file.find('.py')]
        #         importlib.import_module('optim.' + module)
    else:
        device = torch.device('cpu')
    print("Running on device {}".format(device))
    args.device = device
    
    if args.max_tokens is None:
        args.max_tokens = 6000

    torch.manual_seed(args.seed)

    src_dict, tgt_dict = data_utils.load_dictionaries(args)
    datasets = load_dataset_splits(args, ['train', 'valid', 'test'], src_dict, tgt_dict)

    seed = _gen_seeds(seed_shape)
    seed = torch.from_numpy(seed)
    seed = seed.to(device)
    model = build_model(args, seed=seed)
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))

    # optimizer = optim.build_optimizer(args, model.parameters())
    # Build trainer
    trainer = DDPTrainer(args, model)
    print('| model {}, criterion {}'.format(args.arch, trainer.criterion.__class__.__name__))

    print('| training on {} NPUs'.format(args.distributed_world_size))
    print('| max sentences per NPU = {}'.format(args.max_sentences))

    writer = SummaryWriter(args.save_dir)

    epoch_itr = data.EpochBatchIterator(
        dataset=datasets[args.train_subset],
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=args.max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=1,
        shard_id=0,
        max_positions_num=96,

    )

    # Train until the learning rate gets too small or model reaches target score
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    run_summary = {'loss': float('inf'),
                   'val_loss': float('inf'),
                   'speed': 0,
                   'accuracy': 0}

    # max_update
    while lr >= args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        
        # train for one epoch
        train(args, trainer, datasets, epoch_itr)

        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, datasets, valid_subsets)

        writer.add_scalar('loss/val', valid_losses[0], epoch_itr.epoch)

        # Only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])


    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, datasets, epoch_itr):
    """Train the model for one epoch."""

    itr = epoch_itr.next_epoch_itr()

    # update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    num_batches = len(epoch_itr)

    batch_time = AverageMeter('Time', ':6.3f')
    sentence_s = AverageMeter('Sentence/s', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')

    progress = ProgressMeter(int(num_batches/update_freq),
                             [batch_time, sentence_s,losses],
                             prefix = "Epoch: [{}]".format(epoch_itr.epoch))

    print("Update Frequence is :", str(update_freq))

    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf

    trainer.get_throughput_meter().reset()

    for i, sample in enumerate(itr):
        if i < num_batches - 1 and (i + 1) % update_freq > 0:
            # buffer updates according to --update-freq
            loss = trainer.train_step(sample, update_params=False, last_step=(i == len(itr) - 1))
            continue
        else:
            loss = trainer.train_step(sample, update_params=True, last_step=(i == len(itr) - 1))
            if loss != None:
                losses.update(loss)
        if i >= 10:
            t = time.time()
            batch_time.update((t - end)/update_freq)
            sentence_s.update(args.max_sentences/(t-end)*update_freq)
            end = time.time()
        if i < 10:
            end = time.time()
        if i >= 10:
            progress.display(int((i+1)/update_freq))


        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_throughput_meter().reset()

        # Mid epoch checkpoint
        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0:
            valid_losses = validate(args, trainer, datasets, [first_valid])
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    print("End of epoch, batch_size:", args.max_sentences, 'Time: {:.3f}'.format(batch_time.avg), ' Sentence/s@all {:.3f}'.format(
        args.max_sentences / batch_time.avg))


def validate(args, trainer, datasets, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    # Reset value iterations counter
    trainer._num_val_iterations = 0

    valid_losses = []
    for subset in subsets:

        if len(subsets) > 1:
            print('Validating on \'{}\' subset'.format(subset))

        # Initialize data iterator
        itr = data.EpochBatchIterator(
            dataset=datasets[subset],
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=args.max_positions,
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=1,
            shard_id=0,
            max_positions_num=1024,
        ).next_epoch_itr(shuffle=False)


        subset_losses = []
        for sample in itr:
            loss = trainer.valid_step(sample)
            subset_losses.append(loss)
        subset_loss = sum(subset_losses) / len(subset_losses)

        valid_losses.append(subset_loss)
        print(f'Validation loss on subset {subset}: {subset_loss}')

    return valid_losses


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'best': save_checkpoint.best,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    extra_state.update(save_checkpoint.extra_items)

    checkpoints = [os.path.join(args.save_dir, 'checkpoints', fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(os.path.join(args.save_dir, 'checkpoints'),
                                             pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def add_extra_items_to_checkpoint(dict):
    if not hasattr(save_checkpoint, 'extra_items'):
        save_checkpoint.extra_items = {}
    save_checkpoint.extra_items.update(dict)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoints', args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']


if __name__ == '__main__':
    parser = options.get_training_parser()
    ARGS = options.parse_args_and_arch(parser)

    main(ARGS)