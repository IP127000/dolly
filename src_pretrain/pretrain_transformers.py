import torch
from src_llm.modeling_dolly import DollyModel, DollyForCausalLM
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    set_seed
)
from datasets import load_dataset
from accelerate import Accelerator
import os
import logging

# 设置基本日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化加速器
accelerator = Accelerator()

# 设置随机种子（为了可复现性）
set_seed(42)

# 模型路径
model_name = "/mnt/han.luzhi/weights_llm"  

# 加载tokenizer（添加padding token如果不存在）
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型（使用AutoModelForCausalLM而不是自定义类）
model = DollyForCausalLM.from_pretrained(model_name)

# 启用梯度检查点（节省显存）
model.gradient_checkpointing_enable()

# 加载数据集
dataset = load_dataset('text', data_files={'train': '/mnt/han.luzhi/corpus/wikipedia.txt'})

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )

# 处理数据集
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['text'],  # 移除原始文本列以节省空间
    num_proc=4  # 多进程处理
)

# 使用专门的语言模型数据收集器（会自动处理labels）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 对于因果语言模型设为False
)

# 训练参数
training_args = TrainingArguments(
    output_dir="/mnt/han.luzhi/result",
    do_eval=False,  # 改为按步评估
    save_strategy="steps",  # 与评估同步
    save_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=26,
    num_train_epochs=5,
    logging_dir="/mnt/han.luzhi/logs",
    logging_steps=10,
    save_total_limit=2,
    fp16=True,  # 或者使用bf16=True如果硬件支持
    dataloader_num_workers=4,
    gradient_accumulation_steps=2,  # 累积梯度以减少显存使用
    report_to="tensorboard",  # 添加TensorBoard支持
    optim="adamw_torch",  # 使用PyTorch实现的AdamW
    lr_scheduler_type="cosine",  # 使用cosine学习率调度
    warmup_steps=500,  # 添加warmup
    weight_decay=0.01,
    local_rank=int(os.environ.get("LOCAL_RANK", 0)),  # 直接从环境变量获取
    disable_tqdm=False,  # 确保启用进度条
    logging_first_step=True,  # 启用第一步日志输出
    log_level="info",  # 设置日志级别
    log_level_replica="info",  # 设置副本（GPU）日志级别
    ddp_find_unused_parameters=False,  # 提高分布式训练效率
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    processing_class=tokenizer,  # 这里应该是tokenizer而不是processing_class
    data_collator=data_collator,
)

# 打印开始训练信息
logger.info("***** 开始训练 *****")
logger.info(f" 进程数 = {accelerator.num_processes}")
logger.info(f" 设备 = {accelerator.device}")
logger.info(f" 总批大小 = {training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps}")

# 开始训练
trainer.train()

# 保存最终模型
if accelerator.is_main_process:  # 只在主进程保存
    trainer.save_model("/mnt/han.luzhi/result/final")
    tokenizer.save_pretrained("/mnt/han.luzhi/result/final")
    logger.info("***** 训练完成，模型已保存 *****")

##  python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 pretrain_transformers.py