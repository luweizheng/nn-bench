# May help avoid undefined symbol errors https://pytorch.org/cppdocs/notes/faq.html#undefined-symbol-errors-from-pytorch-aten
import torch
import warnings

if torch.distributed.is_available():
    from apex import parallel

from . import amp
