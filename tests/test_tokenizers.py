from transformers import AutoTokenizer
import time

TOKENIZER_DIR = '../data/tokenizer-sp-hf'
SP_TOKENIZER_DIR = '../data/tokenizer-sp'

text = "Héllò hôw are ü?" * 20

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
start_time = time.perf_counter()
tokens = tokenizer.tokenize(text, add_special_tokens=False)
end_time = time.perf_counter() - start_time
print("HF API SP Tokenizer time ", end_time)
print('Reloaded Tokenizer: ', tokens)

print("bos_token_id", tokenizer.bos_token_id)
print("cls_token_id", tokenizer.cls_token_id)
print("eos_token_id", tokenizer.eos_token_id)
print("mask_token_id", tokenizer.mask_token_id)
print("pad_token_id", tokenizer.pad_token_id)
print("sep_token_id", tokenizer.sep_token_id)
print("unk_token_id", tokenizer.unk_token_id)

sp_tokenizer = AutoTokenizer.from_pretrained(SP_TOKENIZER_DIR)
start_time = time.perf_counter()
tokens = sp_tokenizer.tokenize(text, add_special_tokens=False)
end_time = time.perf_counter() - start_time
print("SP Tokenizer in HF Wrapper (DeBerta) time ", end_time)
print('Reloaded Tokenizer: ', tokens)

print("bos_token_id", sp_tokenizer.bos_token_id)
print("cls_token_id", sp_tokenizer.cls_token_id)
print("eos_token_id", sp_tokenizer.eos_token_id)
print("mask_token_id", sp_tokenizer.mask_token_id)
print("pad_token_id", sp_tokenizer.pad_token_id)
print("sep_token_id", sp_tokenizer.sep_token_id)
print("unk_token_id", sp_tokenizer.unk_token_id)
