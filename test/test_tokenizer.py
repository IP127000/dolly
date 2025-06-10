from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("weights/weights_tokenizer/hf_tokenizer_BBPE")
# tokenizer = AutoTokenizer.from_pretrained("temp")
# 测试特殊标记
print(tokenizer.special_tokens_map)

# 测试分词
text = "Hello World<|im_end|> 你好世界"
print(tokenizer.tokenize(text))

# 测试聊天模板
messages = [{"role": "user", "content": "Hello!"}]
print(tokenizer.apply_chat_template(messages))

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print("\n格式化后的对话:",formatted)
print(tokenizer.vocab_size)