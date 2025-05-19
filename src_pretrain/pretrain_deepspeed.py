import torch
from src_llm.modeling_dolly import DollyForCausalLM
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    set_seed
)
from datasets import load_dataset
from accelerate import Accelerator
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

accelerator = Accelerator()
set_seed(42)

model_name = "/mnt/han.luzhi/dolly_llm/weights_llm"  
# resume_option = None                     
resume_option = True                    
# resume_option = "/mnt/han.luzhi/result/checkpoint-500"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if isinstance(resume_option, str) and os.path.exists(resume_option):
    logger.info(f"从指定检查点 {resume_option} 加载模型...")
    model = DollyForCausalLM.from_pretrained(resume_option)
elif resume_option is True and os.path.exists("/mnt/han.luzhi/dolly_llm/result"):
    latest_checkpoint = Trainer.find_latest_checkpoint("/mnt/han.luzhi/dolly_llm/result")
    if latest_checkpoint:
        logger.info(f"自动发现最新检查点 {latest_checkpoint}...")
        model = DollyForCausalLM.from_pretrained(latest_checkpoint)
    else:
        logger.info("未找到检查点，从预训练模型加载...")
        model = DollyForCausalLM.from_pretrained(model_name)
else:
    logger.info(f"从预训练模型 {model_name} 加载...")
    model = DollyForCausalLM.from_pretrained(model_name)

model.gradient_checkpointing_enable()

dataset = load_dataset('text', data_files={'train': '/mnt/han.luzhi/dolly_llm/corpus/wikipedia.txt'})

def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['text'], 
    num_proc=4  
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False 
)

training_args = TrainingArguments(
    output_dir="/mnt/han.luzhi/dolly_llm/result",
    deepspeed="/mnt/han.luzhi/dolly_llm/deepspeed_config/ds_stage2.json",  
    do_eval=False,
    save_strategy="steps",
    save_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=26,
    num_train_epochs=5,
    logging_dir="/mnt/han.luzhi/dolly_llm/logs",
    logging_steps=10,
    save_total_limit=2,
    fp16=True,  
    dataloader_num_workers=4,
    gradient_accumulation_steps=2,
    report_to="tensorboard",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_steps=500,
    weight_decay=0.01,
    local_rank=int(os.environ.get("LOCAL_RANK", 0)),
    disable_tqdm=False,
    logging_first_step=True,
    log_level="info",
    ddp_find_unused_parameters=False,
    overwrite_output_dir=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    processing_class=tokenizer, 
    data_collator=data_collator,
)

logger.info("***** 开始训练 *****")
logger.info(f" 进程数 = {accelerator.num_processes}")
logger.info(f" 设备 = {accelerator.device}")
logger.info(f" 总批大小 = {training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps}")

trainer.train(resume_from_checkpoint=resume_option)

if accelerator.is_main_process: 
    trainer.save_model("/mnt/han.luzhi/dolly_llm/result/final")
    tokenizer.save_pretrained("/mnt/han.luzhi/dolly_llm/result/final")
    logger.info("***** 训练完成，模型已保存 *****")

##  deepspeed --num_gpus=4 --master_port=12345 pretrain_transformers.py
#   deepspeed --include="localhost:1,2,3" --master_port=12345 pretrain_transformers.py