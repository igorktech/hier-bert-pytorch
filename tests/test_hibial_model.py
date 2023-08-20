from transformers import AlbertTokenizer
from hibial_model.modelling_hibial import HiBiAlBertModel
from hibial_model.configuration_hibial import HiBiAlBertConfig

# Load the pretrained model and tokenizer
model_name = 'albert-base-v2'
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = HiBiAlBertModel(HiBiAlBertConfig())

# Example input
text = "Hello, how are you? [SEP] I am fine thank you. [SEP] How was your weekend? [SEP]"

# Tokenize the input
# tokens = tokenizer.tokenize(text)
# inputs = tokenizer.encode_plus(tokens,
#                                return_tensors='pt',
#                                truncation=True,
#                                add_special_tokens=True,
#                                padding='max_length',
#                                max_length=512,
#                                pad_to_max_length=True, )
sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"
inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')
# Get the input tensors
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
token_type_ids = inputs['token_type_ids']
print('input ids', input_ids)
print('attention mask', attention_mask)
print('token type ids', token_type_ids)
# Forward pass through the model
outputs = model(input_ids=input_ids,  # attention_mask=attention_mask,# token_type_ids=token_type_ids,
                return_dict=False)

# Get the output tensors
sequence_output = outputs[0]
pooled_output = outputs[1]
# Print the shapes of the output tensors
print("Sequence Output Shape:", sequence_output.shape)
print("Pooled Output Shape:", pooled_output.shape)
print(outputs[0])
