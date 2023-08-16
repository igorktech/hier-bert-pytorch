import torch
import seaborn as sns
import matplotlib.pyplot as plt

from hierbert_model.modelling_hierbert import get_hier_encoder_mask

utt_indices = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 3, 3, 3]])
src_input_ids = torch.randint(0, 1, (1, 10))
src_padding_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])

pe_utt_loc, enc_mask_utt, enc_mask_ct = get_hier_encoder_mask(
    src_input_ids,
    src_padding_mask,
    utt_indices,
    type="cls")

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Various HIER-CLS Masks')
sns.heatmap((src_padding_mask.float()[0]).unsqueeze(0).expand(src_input_ids.float().shape[1], -1).cpu().numpy(),
            ax=axes[0, 0]).set_title("SRC Padding Mask")
sns.heatmap((enc_mask_utt[0] * 1).cpu().numpy(), ax=axes[0, 1]).set_title("UT_Mask")
sns.heatmap((enc_mask_ct[0] * 1).cpu().numpy(), ax=axes[1, 0]).set_title("CT_Mask")

plt.show()
