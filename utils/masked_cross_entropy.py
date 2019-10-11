#!/usr/bin/env python
#conding: utf-8

#@Author: Dimmy(wangdimmy@gmail.com)
#@Description: This script is extracted from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1

import torch
import torch.nn.functional as functional
from utils.config import *


def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()

    batch_size = len(sequence_length)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if USE_CUDA:
        seq_range_expand = seq_range_expand.cuda()

    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    if USE_CUDA:
        seq_length_expand = seq_length_expand.cuda()

    return seq_range_expand < seq_length_expand


def compute_loss(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))

    # log_probs_flat: (batch * max_len, num_classes)
    #log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(logits_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(target.size())

    mask = _sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss