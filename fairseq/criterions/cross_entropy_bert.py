# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('cross_entropy_bert')
class CrossEntropyBertCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.enforce_idempotence = args.enforce_idempotence

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.enforce_idempotence:
            mlm_out, nsp_out, mlm_out_0, emb_mask = model(**sample['net_input'])
        else:
            mlm_out, nsp_out = model(**sample['net_input'])

        mlm_lprobs = model.get_normalized_probs(mlm_out, log_probs=True)
        mlm_lprobs = mlm_lprobs.view(-1, mlm_lprobs.size(-1))
        target = model.get_targets(sample, mlm_out)[sample['net_input']['output_mask']].view(-1)
        mlm_loss = F.nll_loss(mlm_lprobs, target, ignore_index=self.padding_idx, reduction='sum' if reduce else 'none')
        mlm_acc = (mlm_lprobs.argmax(dim=-1).eq(target)).float().sum()
        mlm_size = target.ne(self.padding_idx).long().sum().item()

        nsp_out = nsp_out.float()
        label = model.get_label(sample)  # B
        nsp_loss = F.binary_cross_entropy_with_logits(nsp_out, label.float(), reduction='sum' if reduce else 'none')
        nsp_acc = ((nsp_out >= 0.0).eq(label.byte())).float().sum()

        n_sentences = sample['target'].size(0)

        if self.enforce_idempotence:
            mlm_0_lprobs = model.get_normalized_probs(mlm_out_0, log_probs=True)
            mlm_0_lprobs = mlm_0_lprobs.view(-1, mlm_0_lprobs.size(-1))
            source = sample['net_input']['src_tokens'][emb_mask].view(-1)
            mlm_0_loss = F.nll_loss(mlm_0_lprobs, source, ignore_index=self.padding_idx,
                                    reduction='sum' if reduce else 'none')
            mlm_0_acc = (mlm_0_lprobs.argmax(dim=-1).eq(source)).float().sum()
            mlm_0_size = source.ne(self.padding_idx).long().sum().item()

            loss = mlm_loss + nsp_loss * mlm_size / n_sentences + mlm_0_loss * mlm_size / mlm_0_size
        else:
            loss = mlm_loss + nsp_loss * mlm_size / n_sentences

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'mlm_loss': utils.item(mlm_loss.data) if reduce else mlm_loss.data,
            'mlm_acc': utils.item(mlm_acc.data) if reduce else mlm_acc.data,
            'nsp_loss': utils.item(nsp_loss.data) if reduce else nsp_loss.data,
            'nsp_acc': utils.item(nsp_acc.data) if reduce else nsp_acc.data,
            'ntokens': sample['ntokens'],
            'nsentences': n_sentences,
            'sample_size': mlm_size,
        }

        if self.enforce_idempotence:
            logging_output['mlm_0_loss'] = utils.item(mlm_0_loss.data) if reduce else mlm_0_loss.data
            logging_output['mlm_0_acc'] = utils.item(mlm_0_acc.data) if reduce else mlm_0_acc.data
            logging_output['mlm_0_size'] = mlm_0_size

        return loss, mlm_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        mlm_loss_sum = sum(log.get('mlm_loss', 0) for log in logging_outputs)
        mlm_acc_sum = sum(log.get('mlm_acc', 0) for log in logging_outputs)
        nsp_loss_sum = sum(log.get('nsp_loss', 0) for log in logging_outputs)
        nsp_acc_sum = sum(log.get('nsp_acc', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        mlm_0_loss_sum = sum(log.get('mlm_0_loss', 0) for log in logging_outputs)
        mlm_0_acc_sum = sum(log.get('mlm_0_acc', 0) for log in logging_outputs)
        mlm_0_size_sum = sum(log.get('mlm_0_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size,
            'mlm_loss': mlm_loss_sum / sample_size,
            'mlm_acc': mlm_acc_sum / sample_size,
            'nsp_loss': nsp_loss_sum / nsentences,
            'nsp_acc': nsp_acc_sum / nsentences,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if mlm_0_size_sum > 0:
            agg_output['mlm_0_loss'] = mlm_0_loss_sum / mlm_0_size_sum
            agg_output['mlm_0_acc'] = mlm_0_acc_sum / mlm_0_size_sum
            agg_output['mlm_0_size'] = mlm_0_size_sum
        return agg_output
