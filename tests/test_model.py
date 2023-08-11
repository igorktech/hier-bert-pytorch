import torch
import torch.nn as nn
from transformers import  AlbertTokenizer
from hierbert_model.modelling_hierbert import HierBertModel
from hierbert_model.configuration_hierbert import HierBertConfig

# Load the pretrained model and tokenizer
model_name = 'albert-base-v2'
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = HierBertModel(HierBertConfig())

# Example input
text = "Hello, how are you? [SEP] I am fine thank you. [SEP] How was your weekend? [SEP]"

# Tokenize the input
tokens = tokenizer.tokenize(text)
inputs = tokenizer.encode_plus( tokens,
                                return_tensors='pt',
                                truncation=True,
                                add_special_tokens=True,
                                padding='max_length',
                                max_length=512,
                                pad_to_max_length=True,)

# Get the input tensors
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print('input_ids',input_ids)
print('attention mask',attention_mask)
# Forward pass through the model
outputs = model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids = inputs['token_type_ids'], return_dict = False)

# Get the output tensors
sequence_output = outputs[0]
pooled_output = outputs[1]
# Print the shapes of the output tensors
print("Sequence Output Shape:", sequence_output.shape)
print("Pooled Output Shape:", pooled_output.shape)
print(outputs[0])

