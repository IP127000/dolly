
from transformers import AutoTokenizer, set_seed
from datasets import load_dataset, DatasetDict
from accelerate import Accelerator
import os
import logging
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

accelerator = Accelerator()
set_seed(42)

TOKENS_DATA_DIR = "../corpus"
os.makedirs(TOKENS_DATA_DIR, exist_ok=True)

model_name = "../weights/weights_tokenizer"  
resume_option = None                     
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

data_files = []
supported_formats = ['.jsonl', '.parquet', '.txt', '.json', '.csv']
for format in supported_formats:
    data_files.extend(glob.glob(f"../train/*{format}"))

if not data_files:
    logger.error("未找到任何支持的语料文件！支持的格式: .jsonl, .parquet, .txt, .json, .csv")
    exit(1)
    
logger.info(f"找到 {len(data_files)} 个训练语料文件: {data_files}")

file_ext = os.path.splitext(data_files[0])[1].lower()

if file_ext in ['.txt', '.text']:
    logger.info("检测到文本文件格式，使用'text'加载器")
    dataset = load_dataset('text', data_files=data_files, split='train')
    dataset = dataset.map(lambda examples: {'text': examples['text']}, batched=True)
else:
    logger.info(f"检测到结构化文件格式 ({file_ext})，使用自动推断加载器")
    dataset = load_dataset(file_ext.lstrip('.'), data_files=data_files, split='train')

if 'text' not in dataset.column_names:
    logger.error(f"数据集不包含必需的'text'列！实际列名: {dataset.column_names}")
    logger.info("请确保语料文件包含文本内容在'text'字段中")
    exit(1)

def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=1024,
        padding='max_length',
        return_tensors='pt'
    )

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=16,
    desc="Tokenizing dataset"
)

logger.info(f"保存tokenized数据集到: {TOKENS_DATA_DIR}")
DatasetDict({"train": tokenized_datasets}).save_to_disk(TOKENS_DATA_DIR)
logger.info(f"Tokenized数据已保存，大小: {len(tokenized_datasets)} 样本")