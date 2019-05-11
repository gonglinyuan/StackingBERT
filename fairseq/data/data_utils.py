# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import contextlib
import os

import numpy as np


def infer_language_pair(path):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    for filename in os.listdir(path):
        parts = filename.split('.')
        if len(parts) >= 3 and len(parts[1].split('-')) == 2:
            return parts[1].split('-')
    return src, dst


def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.

    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)


def filter_by_size(indices, size_fn, max_positions, raise_exception=False):
    """
    Filter indices based on their size.

    Args:
        indices (List[int]): ordered list of dataset indices
        size_fn (callable): function that returns the size of a given index
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception
            if any elements are filtered. Default: ``False``
    """
    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) <= max_positions
        else:
            return all(a is None or b is None or a <= b
                       for a, b in zip(size_fn(idx), max_positions))

    ignored = []
    itr = collect_filtered(check_size, indices, ignored)
    for idx in itr:
        if len(ignored) > 0 and raise_exception:
            raise Exception((
                'Size of sample #{} is invalid (={}) since max_positions={}, '
                'skip this example with --skip-invalid-size-inputs-valid-test'
            ).format(idx, size_fn(idx), max_positions))
        yield idx

    if len(ignored) > 0:
        print((
            '| WARNING: {} samples have invalid sizes and will be skipped, '
            'max_positions={}, first few sample ids={}'
        ).format(len(ignored), max_positions, ignored[:10]))


def _trunc_sent(sent, cnt):
    trunc_head = np.random.binomial(cnt, 0.5)
    trunc_tail = cnt - trunc_head
    return sent[trunc_head:-trunc_tail]


def truncate_single(sent, max_positions):
    diff = sent.size(0) + 1 - max_positions
    if diff <= 0:
        return sent
    return _trunc_sent(sent, diff)


def truncate_pair(sent1, sent2, max_positions):
    diff = sent1.size(0) + sent2.size(0) + 1 - max_positions
    if diff <= 0:
        return sent1, sent2
    if sent1.size(0) > sent2.size(0):
        to_trunc = min(sent1.size(0) - sent2.size(0), diff)
        sent1 = _trunc_sent(sent1, to_trunc)
        diff -= to_trunc
    elif sent1.size(0) < sent2.size(0):
        to_trunc = min(sent2.size(0) - sent1.size(0), diff)
        sent2 = _trunc_sent(sent2, to_trunc)
        diff -= to_trunc
    if diff <= 0:
        return sent1, sent2
    trunc1 = np.random.binomial(diff, 0.5)
    trunc2 = diff - trunc1
    return _trunc_sent(sent1, trunc1), _trunc_sent(sent2, trunc2)


def batch_by_size(
    indices, num_tokens_fn, max_tokens=None, max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch.
            Default: ``None``
        max_sentences (int, optional): max number of sentences in each
            batch. Default: ``None``
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N. Default: ``1``
    """
    max_tokens = max_tokens if max_tokens is not None else float('Inf')
    max_sentences = max_sentences if max_sentences is not None else float('Inf')
    bsz_mult = required_batch_size_multiple

    batch = []

    def is_batch_full(num_tokens):
        if len(batch) == 0:
            return False
        if len(batch) == max_sentences:
            return True
        if num_tokens > max_tokens:
            return True
        return False

    sample_len = 0
    sample_lens = []
    ignored = []
    for idx in indices:
        sample_lens.append(num_tokens_fn(idx))
        sample_len = max(sample_len, sample_lens[-1])
        num_tokens = (len(batch) + 1) * sample_len
        if is_batch_full(num_tokens):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            yield batch[:mod_len]
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0

        batch.append(idx)

    if len(batch) > 0:
        yield batch
