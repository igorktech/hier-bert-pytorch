from hierbert_model.modelling_hierbert import HierBertModel, HierBert, HierBertForMaskedLM, \
    HierBertForSequenceClassification

from transformers import AutoModel, AutoConfig, \
    AutoModelForMaskedLM, \
    AutoModelForSequenceClassification
from transformers import AlbertTokenizer

from hierbert_model.configuration_hierbert import HierBertConfig

# We will use pretrained AlBert tokenizer
tokenizer_name = 'albert-base-v2'
tokenizer = AlbertTokenizer.from_pretrained(tokenizer_name)

model_name = "igorktech/custom"

# Optional to binding code for config.json before publishing
model = HierBertModel(HierBertConfig())
HierBertConfig.register_for_auto_class()
HierBertModel.register_for_auto_class("AutoModel")
HierBertForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
HierBertForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")

# pretrained_model = HierBert(HierBertConfig())
# model.model.load_state_dict(pretrained_model.state_dict())

model.push_to_hub(model_name)


# Example input
text = "Hello, how are you? [SEP] I am fine thank you. [SEP] How was your weekend? [SEP]"

# Tokenize the input
tokens = tokenizer.tokenize(text)
inputs = tokenizer.encode_plus(tokens,
                               return_tensors='pt',
                               truncation=True,
                               add_special_tokens=True,
                               padding='max_length',
                               max_length=512,
                               pad_to_max_length=True, )

# Get the input tensors
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=inputs['token_type_ids'],
                return_dict=False)

# Get the output tensors
sequence_output = outputs[0]
pooled_output = outputs[1]
# Print the shapes of the output tensors
print("Sequence Output Shape:", sequence_output.shape)
print("Pooled Output Shape:", pooled_output.shape)
print(outputs[0])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, trust_remote_code=True)
outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=inputs['token_type_ids'],
                return_dict=False)
print(outputs)
