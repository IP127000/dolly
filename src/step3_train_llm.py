from modeling_dolly import DollyForCausalLM
from configuration_dolly import DollyConfig
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
import glob
from datasets import load_dataset, DatasetDict
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

accelerator = Accelerator()
set_seed(42)
TOKENS_DATA_DIR = "../corpus"
tokenizer_path="../weights/weights_tokenizer"
model_name = "../weights/weights_llm"  
resume_option = None
first_time=True                     
# resume_option = True                    
# resume_option = "../checkpoints_ds/checkpoint-500" 

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
if first_time:
    model = DollyForCausalLM(DollyConfig()) 
else:
    if isinstance(resume_option, str) and os.path.exists(resume_option):
        logger.info(f"从指定检查点 {resume_option} 加载模型...")
        model = DollyForCausalLM.from_pretrained(resume_option)
    else:
        logger.info("未找到检查点，从预训练模型加载...")
        logger.info(f"从预训练模型 {model_name} 加载...")
        model = DollyForCausalLM.from_pretrained(model_name)
        
model.gradient_checkpointing_enable()

tokenized_datasets = DatasetDict.load_from_disk(TOKENS_DATA_DIR)["train"]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False 
)

training_args = TrainingArguments(
    output_dir="../checkpoints_ds",
    deepspeed="../deepspeed_config/ds_stage2.json",  
    do_eval=False,
    save_strategy="steps",
    save_steps=500,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    logging_dir="../logs",
    logging_steps=20,
    save_total_limit=2,
    fp16=True,  
    dataloader_num_workers=4,
    gradient_accumulation_steps=4,
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
    train_dataset=tokenized_datasets,
    processing_class=tokenizer, 
    data_collator=data_collator,
)

logger.info("***** 开始训练 *****")
logger.info(f" 进程数 = {accelerator.num_processes}")
logger.info(f" 设备 = {accelerator.device}")
logger.info(f" 总批大小 = {training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps}")

if resume_option !=None:
    trainer.train(resume_from_checkpoint=resume_option)
else:
    trainer.train()


if accelerator.is_main_process: 
    trainer.save_model("../result/final")
    tokenizer.save_pretrained("../result/final")
    logger.info("***** 训练完成，模型已保存 *****")

##  deepspeed --num_gpus=4 --master_port=12345 pretrain_transformers.py
#   deepspeed --include="localhost:1,2,3" --master_port=12345 pretrain_transformers.py