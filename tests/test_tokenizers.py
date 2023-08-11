from transformers import AutoTokenizer
import time

TOKENIZER_DIR = '../data/tokenizer-sp-hf'
SP_TOKENIZER_DIR = '../data/tokenizer-sp'

text = "Héllò hôw are ü?" * 20

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
start_time = time.perf_counter()
tokens = tokenizer.tokenize(text, add_special_tokens=False)
end_time = time.perf_counter()-start_time
print("HF API SP Tokenizer time ", end_time)
print('Reloaded Tokenizer: ', tokens)

sp_tokenizer = AutoTokenizer.from_pretrained(SP_TOKENIZER_DIR)
start_time = time.perf_counter()
tokens = sp_tokenizer.tokenize(text, add_special_tokens=False)
end_time = time.perf_counter()-start_time
print("SP Tokenizer in HF Wrapper (DeBerta) time ", end_time)
print('Reloaded Tokenizer: ', tokens)

