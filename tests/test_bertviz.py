from bertviz import head_view, model_view
from transformers import AlbertTokenizer
# from hier_model.modelling_hierbert import HierBertModel
# from hier_model.configuration_hierbert import HierBertConfig
from hibial_model.modelling_hibial import HiBiAlBertModel
from hibial_model.configuration_hibial import HiBiAlBertConfig

# Load the pretrained model and tokenizer
model_name = 'albert-base-v2'
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = HiBiAlBertModel(HiBiAlBertConfig())
# HierBertModel.from_pretrained('../data/hier-bert-i3-mlm', trust_remote_code=True)  # (HierBertConfig())

sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"
inputs = tokenizer.encode_plus(sentence_a, "[CLS] " + sentence_b, return_tensors='pt')
input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids']
attention = model(input_ids, token_type_ids=token_type_ids, output_attentions=True)[-1]
sentence_b_start = token_type_ids[0].tolist().index(1)
input_id_list = input_ids[0].tolist()  # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list)

html_head_view = head_view(attention, tokens, html_action='return')  # , sentence_b_start
html_model_view = model_view(attention, tokens, sentence_b_start, display_mode="light", html_action='return')

with open("../data/plots/model_view.html", 'w') as file:
    file.write(html_model_view.data)

with open("../data/plots/head_view.html", 'w') as file:
    file.write(html_head_view.data)
