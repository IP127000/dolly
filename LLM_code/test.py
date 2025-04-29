from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizer_file/hf_tokenizer")  
print(tokenizer.vocab_size)
