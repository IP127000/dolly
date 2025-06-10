from transformers import AutoTokenizer
wrapped_tokenizer = AutoTokenizer.from_pretrained("/mnt/han.luzhi/dolly/llm_tokenizer")

test_text = "深度学习英文是deep learning。"

encoded = wrapped_tokenizer.encode(test_text)
print("IDs:", encoded)           
decoded_text = wrapped_tokenizer.decode(encoded)
print("Decoded:", decoded_text)     


tokens = wrapped_tokenizer.tokenize(test_text)
print("自定义版tokens:", tokens) 
encoded = wrapped_tokenizer.encode(test_text)

tokenizer = AutoTokenizer.from_pretrained("/mnt/qwen2.5_vl_32B",use_fast=False)
tokens = tokenizer.tokenize(test_text)
print("Qwen3版tokens:", tokens) 
encoded = tokenizer.encode(test_text)

