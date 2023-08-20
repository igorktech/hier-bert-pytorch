import torch
import seaborn as sns
import matplotlib.pyplot as plt

from hier_model.modelling_hier import get_hier_encoder_mask

utt_indices = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 2, 2, 2]])
src_input_ids = torch.randint(0, 1, (1, 10))
src_padding_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])

pe_utt_loc, enc_mask_utt, enc_mask_ct = get_hier_encoder_mask(
    src_input_ids,
    src_padding_mask,
    utt_indices,
    type="cls")

sns.set_theme(style='white')
sns.set(rc={'axes.facecolor': '#ece2f0'})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Various HIER-CLS Masks')

sns.heatmap((src_padding_mask.float()[0]).unsqueeze(0).expand(src_input_ids.float().shape[1], -1).cpu().numpy(),
            ax=axes[0, 0], cmap=sns.color_palette("RdPu", as_cmap=True)).set_title("SRC Padding Mask")
sns.heatmap((pe_utt_loc.float()[0]).unsqueeze(0).expand(src_input_ids.float().shape[1], -1).cpu().numpy(),
            ax=axes[0, 1], cmap=sns.color_palette("RdPu", as_cmap=True)).set_title("PE Utterance")

sns.heatmap((enc_mask_utt[0] * 1).cpu().numpy(), ax=axes[1, 0],
            cmap=sns.color_palette("RdPu", as_cmap=True).reversed()).set_title("UT_Mask")
sns.heatmap((enc_mask_ct[0] * 1).cpu().numpy(), ax=axes[1, 1],
            cmap=sns.color_palette("RdPu", as_cmap=True).reversed()).set_title("CT_Mask")

plt.show()
