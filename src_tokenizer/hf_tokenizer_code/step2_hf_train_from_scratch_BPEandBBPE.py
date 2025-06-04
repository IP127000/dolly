import os
import json
from pathlib import Path
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

def get_training_corpus():
    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if "text" in data:
                    yield data["text"]

# 使用Path对象处理路径
corpus_dir = Path("./corpus")
jsonl_files = list(corpus_dir.glob("**/*.jsonl"))
if not jsonl_files:
    raise ValueError("No training files found")

# 使用Qwen2官方特殊标记
special_tokens = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
]

# 创建BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# 使用Qwen2的normalizer配置
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKC(),
    normalizers.StripAccents(),
])

# 简化预分词器：使用ByteLevel代替多个Split规则
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False,  # 对齐Qwen2配置
    use_regex=True
)

# 移除后处理器（Qwen2不需要特殊后处理）
tokenizer.post_processor = None

# 配置解码器
tokenizer.decoder = decoders.ByteLevel(
    use_regex=True,
    trim_offsets=True,
    add_prefix_space=False  # 对齐Qwen2的add_prefix_space配置
)

# 配置训练器
trainer = trainers.BpeTrainer(
    vocab_size=12800,
    special_tokens=special_tokens,
    min_frequency=2,  # 添加最小频率阈值
    show_progress=True  # 显示进度条
)

print("Training tokenizer...")
tokenizer.train_from_iterator(
    get_training_corpus(),
    trainer=trainer,
    length=sum(1 for _ in get_training_corpus())  # 添加总长度用于进度显示
)

# 创建PreTrainedTokenizerFast时添加Qwen2关键配置
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token=None,  # 对齐Qwen2配置
    eos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
    unk_token=None,  # 对齐Qwen2配置
    additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    model_max_length=32768,  # 添加model_max_length
    clean_up_tokenization_spaces=False,  # 对齐Qwen2配置
    errors="replace",  # 对齐Qwen2配置
    split_special_tokens=False,  # 对齐Qwen2配置
)

output_dir = Path("./llm_tokenizer")
output_dir.mkdir(exist_ok=True, parents=True)

# 保存tokenizer
wrapped_tokenizer.save_pretrained(output_dir)
print(f"Tokenizer saved to {output_dir}")

# 使用Qwen2官方chat_template
QWEN2_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}"
    "{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

# 直接设置chat_template避免重新加载
wrapped_tokenizer.chat_template = QWEN2_CHAT_TEMPLATE

# 再次保存包含chat_template的配置
wrapped_tokenizer.save_pretrained(output_dir)

# 测试对话模板
messages = [
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好，有什么可以帮您？"},
    {"role": "user", "content": "Python怎么用？"}
]

formatted = wrapped_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print("\n格式化后的对话:")
print(formatted)

# 测试tokenization
test_text = "深度学习是人工智能的一个重要分支。Deep learning is a subset of machine learning."
print("\nTest tokenization:")
print(wrapped_tokenizer.tokenize(test_text))