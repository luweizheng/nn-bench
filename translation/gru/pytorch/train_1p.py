import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import numpy as np
import math
import argparse
import os
import random
import time
import warnings
import en_core_web_sm
import de_core_news_sm

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
from torchtext.data import Iterator, BucketIterator
from torch.utils.tensorboard import SummaryWriter
from apex import amp

from model import Encoder
from model import Decoder
from model import Seq2Seq

# dataset
spacy_en = en_core_web_sm.load()
spacy_de = de_core_news_sm.load()

# hyperparameter
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
CLIP = 1

MAX = 2147483647


def gen_seeds(num):
    return torch.randint(1, MAX, size=(num,), dtype=torch.float)


seed_init = 0

parser = argparse.ArgumentParser(description='PyTorch Seq2seq-GRU Training')
parser.add_argument('--platform', type=str, default='npu',
                    help='set which type of device you want to use. gpu/npu')
parser.add_argument('--device-id', type=str, default='0',
                    help='set device id')
parser.add_argument('--data', type=str, default='./data', help='path to dataset')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--save-dir', default="./output/", type=str,
                    help='model output path like TensorBoard log.')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=1234, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--print-freq', default=10, type=int,
                    help='frequency to print. ')
# apex
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default=-1, type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--amp-level', default='O2', type=str,
                    help='amp optimization level, O2 means almost FP16, O0 means FP32')


def main():
    args = parser.parse_args()
    print(args)
    if args.platform == "gpu":
        device = torch.device('cuda:' + args.device_id)
        device_func = torch.cuda
    elif args.platform == "npu":
        device = torch.device('npu:' + args.device_id)
        device_func = torch.npu
    else:
        device = torch.device('cpu')
    print("Running on device {}".format(device))
    global CALCULATE_DEVICE
    CALCULATE_DEVICE = device

    main_worker(args)


def main_worker(args):
    # parpare dataset
    SRC = Field(tokenize=tokenize_de,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True, fix_length=46)

    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True, fix_length=46)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(SRC, TRG),
                                                        path=args.data)
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    device = CALCULATE_DEVICE

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=args.batch_size,
        device=device)

    seed_init = gen_seeds(32 * 1024 * 12).float().to(CALCULATE_DEVICE)

    writer = SummaryWriter(args.save_dir)

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT, platform=args.platform, seed=seed_init).to(CALCULATE_DEVICE)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT, platform=args.platform, seed=seed_init).to(CALCULATE_DEVICE)

    model = Seq2Seq(enc, dec, device).to(CALCULATE_DEVICE)
    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = optim.Adam(model.parameters())
    if args.amp:
        print(f"amp_level {args.amp_level}, loss_scale {args.loss_scale}")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_level, loss_scale=args.loss_scale, verbosity=0)
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX).to(CALCULATE_DEVICE)
    best_valid_loss = float('inf')

    for epoch in range(args.epochs):
        bleu_score = calculate_bleu(valid_data, SRC, TRG, model, device)
        writer.add_scalar('bleu/val', bleu_score, epoch)
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, args, CLIP, epoch)
        writer.add_scalar('loss/train', train_loss, epoch)
        valid_loss = evaluate(model, valid_iterator, criterion, args)
        writer.add_scalar('loss/val', valid_loss, epoch)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print(f'\t Val. BLEU score: {bleu_score * 100:.4f}')

    writer.close()
    # load model
    # model.load_state_dict(torch.load('seq2seq-gru-model.pth.tar'))
    # test_loss = evaluate(model, test_iterator, criterion, args)
    # print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    # print(f'BLEU score = {bleu_score * 100:.4f}')


def train(model, iterator, optimizer, criterion, args, clip, epoch):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(iterator),
                             [batch_time, data_time, losses],
                             prefix="Epoch: [{}]".format(epoch))
    epoch_loss = 0

    end = time.time()
    for i, batch in enumerate(iterator):

        data_time.update(time.time() - end)

        src = batch.src.to(CALCULATE_DEVICE)
        trg = batch.trg.to(CALCULATE_DEVICE)

        optimizer.zero_grad()

        output = model(src, trg).to(CALCULATE_DEVICE)

        output_dim = output.shape[-1]
        if args.platform == "npu":
            trg = trg.to(torch.int32)

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        losses.update(loss.item())
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        batch_time.update(time.time() - end)
        # if i % args.print_freq == 0
        #     progress.display(i)

        epoch_loss += loss.item()
        end = time.time()
    print("[device id:", args.device_id, "]",'* FPS@all {:.3f}'.format(args.batch_size / batch_time.avg))
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, args):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src.to(CALCULATE_DEVICE)
            trg = batch.trg.to(CALCULATE_DEVICE)

            output = model(src, trg, 0).to(CALCULATE_DEVICE)  # turn off teacher forcing

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            if args.platform == "npu":
                trg = trg.to(torch.int32)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()

    # tokenize input
    if isinstance(sentence, str):
        nlp = de_core_news_sm.load()
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    # add <sos> and <eos>
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    # get input's one-hot vec
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    # add a batch dim and convert into tensor
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_outputs = model.encoder(src_tensor)

    hidden = encoder_outputs

    # get first decoder input (<sos>)'s one hot
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:]


def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):
    trgs = []
    pred_trgs = []

    for datum in data:
        src = vars(datum)['src']
        trg = vars(datum)['trg']

        pred_trg = translate_sentence(src, src_field, trg_field, model, device, max_len)

        # cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = 5

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        if self.count > (self.start_count_index * n):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * n)

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
        print("[device id:", '0', "]", '\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == "__main__":
    main()