"""
Copyright (c) 2021 by Bishal Santra

All rights reserved.
This file is part of the hier-transformer,
and is released under the "MIT License Agreement". Please see the LICENSE
file that should have been included as part of this package.
"""

import torch


def gen_encoder_ut_mask(src_seq, input_mask, utt_loc):
    def _gen_mask_hierarchical(A, src_pad_mask):
        # A: (bs, 100, 100); 100 is max_len*2 same as input_ids
        return ~(2 * A == (A + A.transpose(1, 2))).bool()

    enc_mask_utt = _gen_mask_hierarchical(utt_loc.unsqueeze(1).expand(-1, src_seq.shape[1], -1), input_mask)
    return enc_mask_utt


def _get_pe_inputs(src_seq, input_mask, utt_loc):
    pe_utt_loc = torch.zeros(utt_loc.shape, device=utt_loc.device)
    for i in range(1, utt_loc.shape[1]):  # time
        _logic = (utt_loc[:, i] == utt_loc[:, i - 1]).float()
        pe_utt_loc[:, i] = pe_utt_loc[:, i - 1] + _logic - (1 - _logic) * pe_utt_loc[:, i - 1]
    return pe_utt_loc


def _CLS_masks(src_seq, input_mask, utt_loc):
    # HT-Encoder
    pe_utt_loc = _get_pe_inputs(src_seq, input_mask, utt_loc)

    # UT-MASK
    enc_mask_utt = gen_encoder_ut_mask(src_seq, input_mask, utt_loc)

    # CT-MASK
    enc_mask_ct = ((pe_utt_loc + input_mask) != 0).unsqueeze(1).expand(-1, src_seq.shape[1], -1)  # HIER-CLS style

    return pe_utt_loc, enc_mask_utt, enc_mask_ct


def get_hier_encoder_mask(src_seq, input_mask, utt_loc, type: str):
    # Padding correction
    # No token other than padding should attend to padding
    # But padding needs to attend to padding tokens for numerical stability reasons
    utt_loc = utt_loc - 2 * input_mask * utt_loc

    # CT-Mask type
    assert type in ["hier", "cls", "full"]

    if type == "hier":  # HIER: Context through final utterance
        raise Exception("Not used for BERT")
    elif type == "cls":  # HIER-CLS: Context through cls tokens
        return _CLS_masks(src_seq, input_mask, utt_loc)
    elif type == "full":  # Ut-mask only, CT-mask: Full attention
        raise Exception("Not used for BERT")

    return None
