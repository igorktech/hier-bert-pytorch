import copy
import math
from typing import Optional, Any, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers import PreTrainedModel
from transformers import BertForMaskedLM, BertForSequenceClassification

from .configuration_hier import HierBertConfig

import warnings

# Turn off all warnings
warnings.filterwarnings("ignore")


# Define masking
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


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class PositionalEmbedding(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.max_len = config.max_position_embeddings
        self.d_model = config.hidden_size
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.max_len, self.d_model).float()
        pe.require_grad = False

        position = torch.arange(0, self.max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Shape of X : [N x L x d] or [N x L]
        return self.pe[:, :x.size(1)]

    def forward_by_index(self, loc):
        return self.pe.expand(loc.shape[0], -1, -1).gather(1, loc.unsqueeze(2).expand(-1, -1, self.pe.shape[2]).long())


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(config.hidden_size,
                                            config.num_attention_heads,
                                            dropout=config.attention_probs_dropout_prob)
        # Implementation of Feedforward model
        self.linear1 = Linear(config.hidden_size, config.intermediate_size)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.linear2 = Linear(config.intermediate_size, config.hidden_size)

        self.norm_first = config.norm_first
        self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout1 = Dropout(config.hidden_dropout_prob)
        self.dropout2 = Dropout(config.hidden_dropout_prob)

        self.activation = _get_activation_fn(config.hidden_act)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        # Extend mask
        # src_mask = src_mask.repeat(self.self_attn.num_heads, 1, 1)

        # PreLayerNorm
        if self.norm_first:

            src = self.norm1(src)
            src_attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask, average_attn_weights=False)  # [0]
            src = src + self.dropout1(src_attn[0])
            src = self.norm2(src)
            src_ffn = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src_ffn)

        else:
            src_attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask, average_attn_weights=False)  # [0]
            src = src + self.dropout1(src_attn[0])
            src = self.norm1(src)
            src_ffn = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src_ffn)
            src = self.norm2(src)
        return src, src_attn[1]


class HierBert(Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
    Examples::
        # >>> transformer_model = HIERTransformer(nhead=16, num_encoder_layers=12)
        # >>> src = torch.rand((10, 32, 512))
        # >>> token_type_ids/utt_indices = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3]) Represent each utterance to encode
        # >>> out = transformer_model(src)
    Note: A full example to apply nn.Transformer module for the word language model is available in
    # https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, config) -> None:
        super(HierBert, self).__init__()
        self.config = config
        # Word Emb
        self.word_embeddings = torch.nn.Embedding(config.vocab_size,
                                                  config.hidden_size,
                                                  padding_idx=config.pad_token_id)

        # Pos Emb
        self.post_word_emb = PositionalEmbedding(config)

        # Encoder
        self.enc_layers = _get_clones(TransformerEncoderLayer(config=config),
                                      config.num_hidden_layers)
        self.norm_e = LayerNorm(config.hidden_size,
                                eps=config.layer_norm_eps)

        self._reset_parameters()
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)

    # TODO: fix return dict
    def forward(self, input_ids: Tensor,
                attention_mask: Optional[Tensor] = None,
                token_type_ids: Optional[Tensor] = None,
                ct_mask_type: str = "cls",
                output_attentions: Optional[bool] = True,
                memory_key_padding_mask: Optional[Tensor] = None,
                **kwargs
                ):
        r"""Take in and process masked source/target sequences.
        Args:
            input_ids/src: the sequence to the encoder (required).
            src_mask: the additive mask for the src sequence (optional).

            memory_mask: the additive mask for the encoder output (optional).
            attention_mask/src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).

            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).
        Shape:
            - input_ids/src: :math:`(S, N, E)`.
            - src_mask: :math:`(S, S)`.
            - memory_mask: :math:`(T, S)`.
            - not(attention_mask)/src_key_padding_mask: :math:`(N, S)`.
            - token_type_ids/utt_indices: :math:`(N, S)`.
            - memory_key_padding_mask: :math:`(N, S)`.
            Note: [src/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - output: :math:`(T, N, E)`.
            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.
            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number
        Examples:
            # >>> output = transformer_model(src, src_mask=src_mask)
        """
        all_self_attentions = () if output_attentions else None
        # print(input_ids.shape)

        if attention_mask is None:
            # Convert input_ids to attention mask
            attention_mask = self.create_padding_mask(input_ids)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        if token_type_ids is None:
            # Convert input_ids to token type IDs
            token_type_ids = self.convert_input_ids_to_token_type_ids(input_ids)
            print('token type ids model', token_type_ids)

        src_key_padding_mask = torch.logical_not(attention_mask)
        utt_indices = token_type_ids

        pe_utt_loc, enc_mask_utt, enc_mask_ct = get_hier_encoder_mask(input_ids,
                                                                      src_key_padding_mask,
                                                                      utt_indices,
                                                                      type=ct_mask_type)

        # memory = self.encoder(input_ids, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Encoding
        # memory = input_ids

        enc_inp = self.word_embeddings(input_ids.transpose(0, 1)) + self.post_word_emb.forward_by_index(
            pe_utt_loc).transpose(0, 1)

        # Basic config
        # for i, layer in enumerate(self.enc_layers):
        #     if i == self.config.num_hidden_layers // 2:
        #         # Positional Embedding for Context Encoder
        #         enc_inp = enc_inp + self.post_word_emb(enc_inp.transpose(0, 1)).transpose(0, 1)
        #     if i < self.config.num_hidden_layers // 2:
        #         enc_inp = layer(enc_inp,
        #                         src_key_padding_mask=src_key_padding_mask,
        #                         src_mask=enc_mask_utt.float())
        #     else:
        #         enc_inp = layer(enc_inp,
        #                         src_key_padding_mask=src_key_padding_mask,
        #                         src_mask=enc_mask_ct)

        # TODO: add layers configurations support and variations setup
        # interleaved config (I3)
        for i, layer in enumerate(self.enc_layers):
            if i % (2 + 1) < 2:
                # Shared encoders or Segment-wise encoders
                # print("SWE")
                enc_inp, att_w = layer(enc_inp,
                                       # src_key_padding_mask=src_key_padding_mask,
                                       src_mask=enc_mask_utt.repeat(self.config.num_attention_heads, 1, 1))
            else:
                # Positional Embedding for Context Encoder if few connected CSE  use it before
                enc_inp = enc_inp + self.post_word_emb(enc_inp.transpose(0, 1)).transpose(0, 1)
                # Context encoder or Cross-segment encoders
                # print("CSE")
                enc_inp, att_w = layer(enc_inp,
                                       # src_key_padding_mask=src_key_padding_mask,
                                       src_mask=enc_mask_ct.repeat(self.config.num_attention_heads, 1, 1))
            if output_attentions:
                all_self_attentions = all_self_attentions + (att_w,)

        if self.norm_e is not None:
            enc_inp = self.norm_e(enc_inp)

        encoder_output = enc_inp.transpose(0, 1)
        hidden_states = encoder_output

        pooled_output = hidden_states[:, 0, :]
        outputs = (hidden_states, pooled_output, all_self_attentions)

        return outputs

    def create_padding_mask(self, token_ids):
        padding_mask = torch.ne(token_ids, self.config.pad_token_id).int()
        return padding_mask

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def convert_input_ids_to_token_type_ids(self, input_ids):
        token_type_ids = torch.zeros_like(input_ids)

        for row, row_tensor in enumerate(input_ids):
            sep_indices = torch.nonzero(row_tensor == self.config.sep_token_id)
            prev_index = -1
            for type_id, index in enumerate(sep_indices):
                token_type_ids[row, prev_index + 1:index + 1] = type_id
                prev_index = index

        return token_type_ids


class HierBertModel(PreTrainedModel):
    config_class = HierBertConfig
    base_model_prefix = "hier"

    def __init__(self, config):
        super().__init__(config)

        self.model = HierBert(config)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             inputs_embeds=inputs_embeds,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states,
                             return_dict=return_dict)
        if not return_dict:
            return outputs

        return BaseModelOutputWithPooling(
            last_hidden_state=outputs[0],
            pooler_output=outputs[1],
            attentions=outputs[2])

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value


class HierBertForMaskedLM(BertForMaskedLM):
    config_class = HierBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = HierBertModel(config)


class HierBertForSequenceClassification(BertForSequenceClassification):
    config_class = HierBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = HierBertModel(config)
