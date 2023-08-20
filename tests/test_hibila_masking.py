import torch
import seaborn as sns
import matplotlib.pyplot as plt
from hibial_model.modelling_hibial import get_hier_encoder_mask
from hibial_model.modelling_hibial import BiALiBi
import numpy as np

utt_indices = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 2, 2, 2]])
src_input_ids = torch.randint(0, 1, (1, 10))
src_padding_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])

from dataclasses import dataclass


@dataclass
class Conf:
    num_attention_heads = 1

config = Conf()
bialibi_utt = BiALiBi(config)
bialibi_ct = BiALiBi(config)

pe_utt_loc, enc_mask_utt, enc_mask_ct = get_hier_encoder_mask(
    src_input_ids,
    src_padding_mask,
    utt_indices,
    type="cls")

enc_mask_utt = enc_mask_utt.repeat(config.num_attention_heads, 1, 1)
enc_mask_ct = enc_mask_ct.repeat(config.num_attention_heads, 1, 1)

bialibi_utt_mask = bialibi_utt(src_input_ids.shape[1])
bialibi_ct_mask = bialibi_ct(src_input_ids.shape[1])

bialibi_utt_mask[enc_mask_utt.bool()] = float('-inf')
bialibi_ct_mask[enc_mask_ct.bool()] = float('-inf')

sns.set_theme(style='white')
sns.set(rc={'axes.facecolor': '#ece2f0'})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Various HIER-CLS Masks')

sns.heatmap((src_padding_mask.float()[0]).unsqueeze(0).expand(src_input_ids.float().shape[1], -1).cpu().numpy(),
            ax=axes[0, 0],
            cmap=sns.color_palette("RdPu", 10)).set_title("SRC Padding Mask")
sns.heatmap((pe_utt_loc.float()[0]).unsqueeze(0).expand(src_input_ids.float().shape[1], -1).cpu().numpy(),
            ax=axes[0, 1],
            cmap=sns.color_palette("RdPu", 10)).set_title("PE Utterance")

np_bialibi_utt_mask = (bialibi_utt_mask[0] * 1).detach().numpy()
np_bialibi_ct_mask = (bialibi_ct_mask[0] * 1).detach().numpy()

sns.cubehelix_palette(as_cmap=True)
sns.heatmap(np_bialibi_utt_mask, ax=axes[1, 0], vmin=-2.0,
            cmap=sns.color_palette("RdPu", as_cmap=True)).set_title("UT_Mask")
sns.heatmap(np_bialibi_ct_mask, ax=axes[1, 1], vmin=-2.0,
            cmap=sns.color_palette("RdPu", as_cmap=True)).set_title("CT_Mask")

plt.show()
