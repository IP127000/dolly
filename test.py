import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_from_disk
import deepspeed
import numpy as np
from glob import glob
import logging
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
# 配置参数
MODEL_PATH = "models/qwen2"        # 本地模型路径
TOKENIZED_DATA_DIR = "corpus"  # tokenized数据目录
OUTPUT_DIR = "./output"            # 训练输出目录
DEEPSPEED_CONFIG = "ds_config.json"  # DeepSpeed配置文件
NUM_GPUS = 4                       # GPU数量
PER_DEVICE_BATCH_SIZE = 4          # 每张GPU的batch size
GRADIENT_ACCUM_STEPS = 8           # 梯度累积步数
MAX_TRAIN_STEPS = 100000           # 最大训练步数
SAVE_STEPS = 2000                  # 保存间隔步数
LOGGING_STEPS = 50                 # 日志记录间隔
LEARNING_RATE = 5e-5               # 学习率
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ===== 1. 准备模型和分词器 =====
logger.info("加载模型和分词器...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/d/VMshare/LLM-from-scratch/weights/weights_tokenizer")
file_path="tokenized_data/tokenized_train-00001-of-00399.parquet"
table = pq.read_table(file_path)
input_ids = table["input_ids"].to_numpy()
attention_mask = table["attention_mask"].to_numpy()
print(len(input_ids))
for input_id in input_ids:
    print(input_id)
    print(len(input_id))
    res=tokenizer.decode(input_id)
    # print(res)
    break

file_path2="/mnt/d/VMshare/LLM-from-scratch/corpus/parquet/train-00001-of-00399.parquet"
table2 = pq.read_table(file_path2)
text = table2["text"]
print(len(text))
for value in text:
    print(value)
    break