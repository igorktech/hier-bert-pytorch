import copy
import math
from typing import Optional, Any

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

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .hier_masks import get_hier_encoder_mask


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class PositionalEmbedding(torch.nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

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

        def __init__(self, d_model, nhead, dim_feedforward=3072, dropout=0.1, activation="gelu", layer_norm_eps=1e-5,
                     norm_first=True):
            super(TransformerEncoderLayer, self).__init__()

            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
            # Implementation of Feedforward model
            self.linear1 = Linear(d_model, dim_feedforward)
            self.dropout = Dropout(dropout)
            self.linear2 = Linear(dim_feedforward, d_model)

            self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
            self.dropout1 = Dropout(dropout)
            self.dropout2 = Dropout(dropout)

            self.activation = _get_activation_fn(activation)

            self.norm_first = norm_first

        def __setstate__(self, state):
            if 'activation' not in state:
                state['activation'] = F.relu
            super(TransformerEncoderLayer, self).__setstate__(state)

        def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
            r"""Pass the input through the encoder layer.
            Args:
                src: the sequence to the encoder layer (required).
                src_mask: the mask for the src sequence (optional).
                src_key_padding_mask: the mask for the src keys per batch (optional).
            Shape:
                see the docs in Transformer class.
            """
            if self.norm_first:
                src_mask = src_mask.repeat(self.self_attn.num_heads, 1, 1)
                src = self.norm1(src)
                src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)[0]
                src = src + self.dropout1(src2)
                src = self.norm2(src)
                src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
                src = src + self.dropout2(src2)
            else:
                src_mask = src_mask.repeat(self.self_attn.num_heads, 1, 1)
                src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)[0]
                src = src + self.dropout1(src2)
                src = self.norm1(src)
                src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
                src = src + self.dropout2(src2)
                src = self.norm2(src)

            return src


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

    def __init__(self, d_model, nhead, dim_feedforward=3072, dropout=0.1, activation="gelu", layer_norm_eps=1e-5,
                 norm_first=True):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.norm_first = norm_first

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        if self.norm_first:
            src_mask = src_mask.repeat(self.self_attn.num_heads, 1, 1)
            src = self.norm1(src)
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            src_mask = src_mask.repeat(self.self_attn.num_heads, 1, 1)
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class HIERBERTTransformer(Module):
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

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 d_word_vec: int = 512, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "gelu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, vocab_size: int = 2, pad_index: int = 0,
                 sep_token_id: int = 102) -> None:  # ,pred_outs=True
        super(HIERBERTTransformer, self).__init__()

        # Word Emb
        self.word_emb = torch.nn.Embedding(vocab_size, d_word_vec, padding_idx=pad_index)

        # Pos Emb
        self.post_word_emb = PositionalEmbedding(d_model=d_word_vec)

        # Encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.enc_layers = _get_clones(encoder_layer, num_encoder_layers)  # ModuleList
        self.num_layers_e = num_encoder_layers
        self.norm_e = encoder_norm

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.pad_index = pad_index
        self.sep_token_id = sep_token_id
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.word_emb.weight.data.uniform_(-initrange, initrange)
        # self.pred_outs=pred_outs

        # self.config.use_return_dict = False

    # TODO: fix return dict
    def forward(self, input_ids: Tensor,
                attention_mask: Optional[Tensor] = None,
                token_type_ids: Optional[Tensor] = None,
                ct_mask_type: str = "cls",
                memory_key_padding_mask: Optional[Tensor] = None,
                labels=None,  # mlm
                next_sentence_label=None,  # nsp
                return_dict=True,
                **kwargs
                ) -> Tensor:
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

        src = input_ids

        if attention_mask is None:
            # Convert input_ids to attention mask
            attention_mask = self.create_padding_mask(input_ids.tolist())
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        if token_type_ids is None:
            # Convert input_ids to token type IDs
            token_type_ids = self.convert_input_ids_to_token_type_ids(input_ids)
            # print(token_type_ids.shape)

        src_key_padding_mask = torch.logical_not(attention_mask)
        utt_indices = token_type_ids
        # print(src.shape, src_key_padding_mask.shape,utt_indices.shape)
        pe_utt_loc, enc_mask_utt, enc_mask_ct = get_hier_encoder_mask(src,
                                                                      src_key_padding_mask,
                                                                      utt_indices,
                                                                      type=ct_mask_type)

        # memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Encoding
        # memory = src

        enc_inp = self.word_emb(src.transpose(0, 1)) + self.post_word_emb.forward_by_index(pe_utt_loc).transpose(0, 1)

        for i, layer in enumerate(self.enc_layers):
            if i == self.num_layers_e // 2:
                # Positional Embedding for Context Encoder
                enc_inp = enc_inp + self.post_word_emb(enc_inp.transpose(0, 1)).transpose(0, 1)
            if i < self.num_layers_e // 2:
                enc_inp = layer(enc_inp,
                                src_key_padding_mask=src_key_padding_mask,
                                src_mask=enc_mask_utt.float())
            else:
                enc_inp = layer(enc_inp,
                                src_key_padding_mask=src_key_padding_mask,
                                src_mask=enc_mask_ct)

        if self.norm_e is not None:
            enc_inp = self.norm_e(enc_inp)

        encoder_output = enc_inp.transpose(0, 1)
        hidden_states = encoder_output

        pooled_output = hidden_states[:, 0, :]
        outputs = (hidden_states, pooled_output)
        if not return_dict:
            return outputs
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output)
        # #This to deal with future problems
        # if not self.pred_outs:
        #     return outputs
        # else:
        #     token_predictions = self.token_prediction_layer(hidden_states)

        #     prediction_scores, seq_relationship_score = self.softmax(token_predictions), self.classification_layer(pooled_output)

        #     total_loss = None
        #     if labels is not None and next_sentence_label is not None:
        #         loss_fct = torch.nn.CrossEntropyLoss()
        #         masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        #         next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
        #         total_loss = masked_lm_loss + next_sentence_loss

        #     if not return_dict:
        #         output = (prediction_scores, seq_relationship_score) + outputs[2:]
        #         return ((total_loss,) + output) if total_loss is not None else output
        #     #TODO
        #     return outputs

    def create_padding_mask(self, token_ids):
        padding_mask = token_ids.eq(self.pad_index)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # Add extra dimensions for broadcasting
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
        input_ids_tensor =  input_ids
        token_type_ids = torch.zeros_like(input_ids_tensor)

        sep_indices = torch.nonzero(input_ids_tensor == self.sep_token_id)

        # Increment the token type ID after each sep token
        for row, row_tensor in enumerate(sep_indices):
            prev_index = -1
            for type_id, index in enumerate(row_tensor):
                token_type_ids[row, prev_index + 1:index + 1] = torch.tensor([type_id] * (index - prev_index))
                prev_index = index

        return token_type_ids
