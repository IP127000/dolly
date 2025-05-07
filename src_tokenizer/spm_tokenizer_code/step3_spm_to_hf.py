from transformers import AutoTokenizer,XLMRobertaTokenizer
#由spm转为hf_tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("tokenizer_file/dolly_tokenizer_spm/tokenizer.model")
text = "A Frenchman from Paris"
encoded_input = tokenizer.encode(text)
print(encoded_input)
decoded_output = tokenizer.decode(encoded_input)
print(decoded_output)
tokenizer.save_pretrained('tokenizer_file/dolly_tokenizer_hf')

#转换为常见的tokenzier.json格式
tokenizer = AutoTokenizer.from_pretrained("tokenizer_file/dolly_tokenizer_hf")
text = "A Frenchman from Paris"
encoded_input = tokenizer.encode(text)
print(encoded_input)
decoded_output = tokenizer.decode(encoded_input)
print(decoded_output)
tokenizer.save_pretrained('tokenizer_file/dolly_tokenizer_hf2')