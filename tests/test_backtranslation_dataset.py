# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import argparse
import unittest

import tests.utils as test_utils
import torch
from fairseq.data.backtranslation_dataset import BacktranslationDataset


class TestBacktranslationDataset(unittest.TestCase):
    def setUp(self):
        self.tgt_dict, self.w1, self.w2, self.src_tokens, self.src_lengths, self.model = (
            test_utils.sequence_generator_setup()
        )
        backtranslation_args = argparse.Namespace()

        """
        Same as defaults from fairseq/options.py
        """
        backtranslation_args.backtranslation_unkpen = 0
        backtranslation_args.backtranslation_sampling = False
        backtranslation_args.backtranslation_max_len_a = 0
        backtranslation_args.backtranslation_max_len_b = 200
        backtranslation_args.backtranslation_beam = 2

        self.backtranslation_args = backtranslation_args

        dummy_src_samples = self.src_tokens

        self.tgt_dataset = test_utils.TestDataset(data=dummy_src_samples)

    def test_backtranslation_dataset(self):
        backtranslation_dataset = BacktranslationDataset(
            args=self.backtranslation_args,
            tgt_dataset=self.tgt_dataset,
            tgt_dict=self.tgt_dict,
            backtranslation_model=self.model,
        )
        dataloader = torch.utils.data.DataLoader(
            backtranslation_dataset,
            batch_size=2,
            collate_fn=backtranslation_dataset.collater,
        )
        backtranslation_batch_result = next(iter(dataloader))

        eos, pad, w1, w2 = self.tgt_dict.eos(), self.tgt_dict.pad(), self.w1, self.w2

        # Note that we sort by src_lengths and add left padding, so actually
        # ids will look like: [1, 0]
        expected_src = torch.LongTensor([[w1, w2, w1, eos], [pad, pad, w1, eos]])
        expected_tgt = torch.LongTensor([[w1, w2, eos], [w1, w2, eos]])
        generated_src = backtranslation_batch_result["net_input"]["src_tokens"]
        tgt_tokens = backtranslation_batch_result["target"]

        self.assertTensorEqual(expected_src, generated_src)
        self.assertTensorEqual(expected_tgt, tgt_tokens)

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertEqual(t1.ne(t2).long().sum(), 0)


if __name__ == "__main__":
    unittest.main()
