import os
import json
from pathlib import Path
import sentencepiece as spm
import logging
import tempfile
import unicodedata
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(message)s')

def normalize_text(text):
    text = text.replace('\x00', '')
    text = unicodedata.normalize('NFKC', text)
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                  if unicodedata.category(c) != 'Mn')
    return text

corpus_dir = Path("./corpus/skypile_2020_head")
jsonl_files = list(corpus_dir.glob("**/*.jsonl"))
if not jsonl_files:
    raise ValueError("No training files found")

special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
logging.info("Staring processing corpus")
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_corpus:
    temp_corpus_path = temp_corpus.name
    file_progress = tqdm(jsonl_files, desc="处理文件中", unit="file")
    for file_path in file_progress:
        file_progress.set_description(f"处理: {file_path.name[:40]}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "text" in data:
                        normalized_text = normalize_text(data["text"])
                        temp_corpus.write(normalized_text + "\n")
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON 解析错误在 {file_path}: {e}")
                except UnicodeDecodeError as e:
                    logging.warning(f"编码错误在 {file_path}: {e}")
logging.info("corpus processing completed")
logging.info("Starting SentencePiece training")

vocab_size = 32000  

# character_coverage = 0.9995  

spm.SentencePieceTrainer.train(
    input=temp_corpus_path,
    model_prefix='tokenizer_model',
    vocab_size=vocab_size,  
    model_type='bpe',
    character_coverage=1.0,  
    pad_id=-1,
    bos_id=-1,
    eos_id=-1,
    unk_id=0,
    user_defined_symbols=special_tokens,
    split_digits=True,
    byte_fallback=True,
    remove_extra_whitespaces=False,
    add_dummy_prefix=False,
    normalization_rule_name='identity',
    num_threads=os.cpu_count(),
    max_sentencepiece_length=32,  
    shuffle_input_sentence=True, 
)

os.unlink(temp_corpus_path)
logging.info("SentencePiece training completed")

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file='tokenizer_model.model',
    bos_token=None,
    eos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
    unk_token=None,
    additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    model_max_length=32768,
    clean_up_tokenization_spaces=False,
    split_special_tokens=False,
    errors="replace"
)

output_dir = Path("./llm_tokenizer")
output_dir.mkdir(exist_ok=True, parents=True)
tokenizer.save_pretrained(output_dir)
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

tokenizer.chat_template = QWEN2_CHAT_TEMPLATE
tokenizer.save_pretrained(output_dir)

messages = [
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好，有什么可以帮您？"},
    {"role": "user", "content": "Python怎么用？"}
]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print("\n格式化后的对话:")
print(formatted)