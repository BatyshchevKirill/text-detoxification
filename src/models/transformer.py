from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


