# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import math

import numpy as np
import torch

from . import data_utils


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count.

    Args:
        iterable (iterable): iterable to wrap

    Attributes:
        count (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable):
        self.iterable = iterable
        self.count = 0
        self.itr = iter(self)

    def __len__(self):
        return len(self.iterable)

    def __iter__(self):
        for x in self.iterable:
            self.count += 1
            yield x

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        """Whether the iterator has been exhausted."""
        return self.count < len(self)

    def skip(self, num_to_skip):
        """Fast-forward the iterator by skipping *num_to_skip* elements."""
        next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
        return self


class EpochBatchIterator(object):
    """A multi-epoch iterator over a :class:`torch.utils.data.Dataset`.

    Compared to :class:`torch.utils.data.DataLoader`, this iterator:

    - can be reused across multiple epochs with the :func:`next_epoch_itr`
      method (optionally shuffled between epochs)
    - can be serialized/deserialized with the :func:`state_dict` and
      :func:`load_state_dict` methods
    - supports sharding with the *num_shards* and *shard_id* arguments

    Args:
        dataset (~torch.utils.data.Dataset): dataset from which to load the data
        collate_fn (callable): merges a list of samples to form a mini-batch
        batch_sampler (~torch.utils.data.Sampler): an iterator over batches of
            indices
        seed (int, optional): seed for random number generator for
            reproducibility. Default: ``1``
        num_shards (int, optional): shard the data iterator into N
            shards. Default: ``1``
        shard_id (int, optional): which shard of the data iterator to
            return. Default: ``0``
    """

    def __init__(self, dataset, collate_fn, batch_sampler, seed=1, num_shards=1, shard_id=0):
        assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.frozen_batches = tuple(batch_sampler)
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id

        self.epoch = 0
        self._cur_epoch_itr = None
        self._next_epoch_itr = None

    def __len__(self):
        return len(self.frozen_batches)

    def next_epoch_itr(self, shuffle=True):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator. Default: ``True``
        """
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self.epoch += 1
            self._cur_epoch_itr = self._get_iterator_for_epoch(self.epoch, shuffle)
        return self._cur_epoch_itr

    def end_of_epoch(self):
        """Returns whether the most recent epoch iterator has been exhausted"""
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        """The number of consumed batches in the current epoch."""
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.count
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.count
        return 0

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        return {
            'epoch': self.epoch,
            'iterations_in_epoch': self.iterations_in_epoch,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict['epoch']
        itr_pos = state_dict.get('iterations_in_epoch', 0)
        if itr_pos > 0:
            # fast-forward epoch iterator
            itr = self._get_iterator_for_epoch(self.epoch, state_dict.get('shuffle', True))
            if itr_pos < len(itr):
                self._next_epoch_itr = itr.skip(itr_pos)

    def _get_iterator_for_epoch(self, epoch, shuffle):
        if shuffle:
            # set seed based on the seed and epoch number so that we get
            # reproducible results when resuming from checkpoints
            with data_utils.numpy_seed(self.seed + epoch):
                batches = list(self.frozen_batches)  # copy
                np.random.shuffle(batches)
        else:
            batches = self.frozen_batches
        return CountingIterator(torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[]),
        ))


class GroupedIterator(object):
    """Wrapper around an iterable that returns groups (chunks) of items.

    Args:
        iterable (iterable): iterable to wrap
        chunk_size (int): size of each chunk
    """

    def __init__(self, iterable, chunk_size):
        self._len = int(math.ceil(len(iterable) / float(chunk_size)))
        self.itr = iter(iterable)
        self.chunk_size = chunk_size

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        chunk = []
        try:
            for _ in range(self.chunk_size):
                chunk.append(next(self.itr))
        except StopIteration as e:
            if len(chunk) == 0:
                raise e
        return chunk


class ShardedIterator(object):
    """A sharded wrapper around an iterable, padded to length.

    Args:
        iterable (iterable): iterable to wrap
        num_shards (int): number of shards to split the iterable into
        shard_id (int): which shard to iterator over
        fill_value (Any, optional): padding value when the iterable doesn't
            evenly divide *num_shards*. Default: ``None``
    """

    def __init__(self, iterable, num_shards, shard_id, fill_value=None):
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError('shard_id must be between 0 and num_shards')

        self._sharded_len = len(iterable) // num_shards
        if len(iterable) % num_shards > 0:
            self._sharded_len += 1

        self.itr = itertools.zip_longest(
            range(self._sharded_len),
            itertools.islice(iterable, shard_id, len(iterable), num_shards),
            fillvalue=fill_value,
        )

    def __len__(self):
        return self._sharded_len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.itr)[1]
