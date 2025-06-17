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
import logging
from tqdm import tqdm
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(), 
        logging.FileHandler(os.path.join(log_dir, "training.log"))  
    ]
)

def get_training_corpus():
    for file_path in tqdm(jsonl_files, desc="processing files", unit="file"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if "text" in data:
                    yield data["text"]


corpus_dir = Path("./corpus/skypile_2020_head")
jsonl_files = list(corpus_dir.glob("**/*.jsonl"))
if not jsonl_files:
    raise ValueError("No training files found")


special_tokens = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
]


tokenizer = Tokenizer(models.BPE())


tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKC(),
    normalizers.StripAccents(),
])


tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Split(r"[\r\n]", "isolated"),   
        pre_tokenizers.Split(r"\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+","isolated"), 
        pre_tokenizers.Split(r"\s?[!-/:-~！-／：-～‘-‟　-。]+", "isolated"),   
        pre_tokenizers.Split(r"\s+$", "isolated"), 
        pre_tokenizers.Split("[一-龥ࠀ-一가-퟿]+","isolated"),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])

# tokenizer.post_processor = None
tokenizer.post_processor = processors.ByteLevel(use_regex=True, trim_offsets=False, add_prefix_space=True)

tokenizer.decoder = decoders.ByteLevel(
    use_regex=True,
    trim_offsets=True,
    add_prefix_space=False  
)


trainer = trainers.BpeTrainer(
    vocab_size=12800,
    special_tokens=special_tokens, 
    show_progress=True  
)

logging.info("Training tokenizer...")
tokenizer.train_from_iterator(
    get_training_corpus(),
    trainer=trainer,
    length=sum(1 for _ in get_training_corpus())  
)

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token=None,  
    eos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
    unk_token=None,  
    additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    model_max_length=32768,  
    clean_up_tokenization_spaces=False,  
    errors="replace", 
    split_special_tokens=False,  
)
logging.info("train done ")
output_dir = Path("./llm_tokenizer")
output_dir.mkdir(exist_ok=True, parents=True)

wrapped_tokenizer.save_pretrained(output_dir)
logging.info(f"Tokenizer saved to {output_dir}")

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

wrapped_tokenizer.chat_template = QWEN2_CHAT_TEMPLATE

wrapped_tokenizer.save_pretrained(output_dir)

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
logging.info("\n格式化后的对话:")
logging.info(formatted)

test_text = "深度学习是人工智能的一个重要分支。Deep learning is a subset of machine learning."
logging.info("\nTest tokenization:")
encoded = wrapped_tokenizer.tokenize(test_text)
logging.info(encoded)
logging.info(encoded)
decoded_text = wrapped_tokenizer.decode(encoded) 
logging.info(decoded_text)
logging.info(decoded_text)