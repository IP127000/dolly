from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("tokenizer_file/hf_tokenizer")

new_tokens = ["<ACTION_1>", "<ACTION_2>"]
tokenizer.add_tokens(new_tokens)

text = "<ACTION_1>"
print(tokenizer.tokenize(text))  
tokenizer.save_pretrained("updated_tokenizer")