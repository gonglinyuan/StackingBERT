# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import FairseqDecoder, FairseqEncoder


class BaseFairseqModel(nn.Module):
    """Base class for fairseq models."""

    def __init__(self):
        super().__init__()
        self._is_generation_fast = False

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def get_label(self, sample):
        return sample['label']

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def max_positions(self):
        """Maximum length supported by the model."""
        return None

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        super().load_state_dict(state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        self.upgrade_state_dict_named(state_dict, '')

    def upgrade_state_dict_named(self, state_dict, name):
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += '.'

            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, 'upgrade_state_dict_named'):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, 'upgrade_state_dict'):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)

    def make_generation_fast_(self, **kwargs):
        """Optimize model for faster generation."""
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True

        # remove weight norm from all modules in the network
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(apply_remove_weight_norm)

        def apply_make_generation_fast_(module):
            if module != self and hasattr(module, 'make_generation_fast_'):
                module.make_generation_fast_(**kwargs)

        self.apply(apply_make_generation_fast_)

        def train(mode):
            if mode:
                raise RuntimeError('cannot train after make_generation_fast')

        # this model should no longer be used for training
        self.eval()
        self.train = train

    def prepare_for_onnx_export_(self, **kwargs):
        """Make model exportable via ONNX trace."""
        def apply_prepare_for_onnx_export_(module):
            if module != self and hasattr(module, 'prepare_for_onnx_export_'):
                module.prepare_for_onnx_export_(**kwargs)

        self.apply(apply_prepare_for_onnx_export_)


class FairseqModel(BaseFairseqModel):
    """Base class for encoder-decoder models.

    Args:
        encoder (FairseqEncoder): the encoder
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing

        Returns:
            the decoder's output, typically of shape `(batch, tgt_len, vocab)`
        """
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())


class FairseqBertModel(BaseFairseqModel):
    def __init__(self, encoder, mid_layer, embed_out, feature_dim, enforce_idempotence=False):
        super().__init__()
        self.encoder = encoder
        self.mid_layer = mid_layer
        self.embed_out = embed_out
        self.classifier = nn.Linear(feature_dim, 1)
        self.enforce_idempotence = enforce_idempotence
        assert isinstance(self.encoder, FairseqEncoder)

    def forward(self, src_tokens, src_lengths, segment, output_mask):
        encoder_return_value = self.encoder(src_tokens, src_lengths, segment, output_mask)
        encoder_out = encoder_return_value['encoder_out']  # (B + N) x C
        mlm_out = F.linear(self.mid_layer(encoder_out[src_tokens.size(0):]), self.embed_out)  # N x V
        nsp_out = self.classifier(encoder_out[:src_tokens.size(0)]).view(-1)  # B
        if self.enforce_idempotence:
            embedding_out = encoder_return_value['embedding_out']  # T x B x C
            # TODO: 1 => padding_idx ; 0.15 => masked_lm_prob
            emb_mask = (src_tokens != 1) & (torch.rand_like(src_tokens, dtype=torch.float) < 0.15)
            mlm_out_0 = F.linear(self.mid_layer(embedding_out.transpose(0, 1)[emb_mask]), self.embed_out)  # M x V
            return mlm_out, nsp_out, mlm_out_0, emb_mask
        else:
            return mlm_out, nsp_out

    def max_positions(self):
        return self.encoder.max_positions()


class FairseqClassifierModel(BaseFairseqModel):
    def __init__(self, encoder, feature_dim, n_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, n_classes)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        assert isinstance(self.encoder, FairseqEncoder)

    def forward(self, src_tokens, src_lengths, segment):
        encoder_out = self.encoder(src_tokens, src_lengths, segment)['encoder_out']  # T x B x C
        features = encoder_out[0, ...]  # B x C
        return self.classifier(features)

    def max_positions(self):
        return self.encoder.max_positions()


class FairseqLanguageModel(BaseFairseqModel):
    """Base class for decoder-only models.

    Args:
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        assert isinstance(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, src_lengths):
        """
        Run the forward pass for a decoder-only model.

        Feeds a batch of tokens through the decoder to predict the next tokens.

        Args:
            src_tokens (LongTensor): tokens on which to condition the decoder,
                of shape `(batch, tgt_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            the decoder's output, typically of shape `(batch, seq_len, vocab)`
        """
        return self.decoder(src_tokens)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.decoder.max_positions()

    @property
    def supported_targets(self):
        return {'future'}
