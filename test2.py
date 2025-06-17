import os
import multiprocessing as mp
from glob import glob
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
import gc
import psutil
import time

# 配置参数
MODEL_PATH = "models/qwen2"  # 本地模型路径
DATA_DIR = "datas"           # 原始数据目录
OUTPUT_DIR = "tokenized_data"  # tokenized数据输出目录
MAX_SEQ_LENGTH = 1024        # 最大序列长度
NUM_PROC = min(32, psutil.cpu_count(logical=False))  # 物理核心数
CHUNK_SIZE = 50000           # 每个处理块的大小
MAX_MEMORY_PERCENT = 70      # 最大内存使用百分比

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("/mnt/d/VMshare/LLM-from-scratch/weights/weights_tokenizer")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 获取所有parquet文件
file_paths = ["/mnt/d/VMshare/LLM-from-scratch/corpus/parquet/train-00001-of-00399.parquet"]
print(f"找到 {len(file_paths)} 个parquet文件")

def memory_safe():
    """检查内存是否安全"""
    mem = psutil.virtual_memory()
    return mem.percent < MAX_MEMORY_PERCENT

def process_chunk(texts, tokenizer):
    """处理文本块并返回tokenized结果"""
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors="np"
    )
    return tokenized["input_ids"], tokenized["attention_mask"]

def process_file(file_path):
    """处理单个大型parquet文件"""
    try:
        base_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_DIR, f"tokenized_{base_name}")
        
        # 如果文件已处理则跳过
        if os.path.exists(output_path):
            print(f"文件已存在: {output_path}, 跳过")
            return
        
        print(f"开始处理: {base_name}")
        start_time = time.time()
        
        # 1. 获取文件总行数
        dataset = load_dataset("parquet", data_files=file_path, split="train")
        total_rows = len(dataset)
        print(f"文件包含 {total_rows:,} 行")
        
        # 2. 创建空DataFrame用于收集结果
        all_input_ids = []
        all_attention_mask = []
        
        # 3. 分块处理
        num_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        for chunk_idx in range(num_chunks):
            if not memory_safe():
                print("内存使用过高，等待释放...")
                time.sleep(30)
                gc.collect()
                if not memory_safe():
                    print("内存仍然不足，跳过当前文件")
                    return
            
            start_row = chunk_idx * CHUNK_SIZE
            end_row = min((chunk_idx + 1) * CHUNK_SIZE, total_rows)
            
            # 读取当前块
            chunk = dataset.select(range(start_row, end_row))
            texts = chunk["text"]  # 根据实际字段名调整
            
            # 处理当前块
            input_ids, attention_mask = process_chunk(texts, tokenizer)
            
            # 收集结果
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            
            # 每处理5个块显示进度
            if (chunk_idx + 1) % 5 == 0 or (chunk_idx + 1) == num_chunks:
                elapsed = time.time() - start_time
                rows_done = min((chunk_idx + 1) * CHUNK_SIZE, total_rows)
                percent = rows_done / total_rows * 100
                print(f"进度: {rows_done:,}/{total_rows:,} 行 ({percent:.1f}%) "
                      f"耗时: {elapsed:.1f}秒")
        
        # 4. 合并所有块
        all_input_ids = np.vstack(all_input_ids)
        all_attention_mask = np.vstack(all_attention_mask)
        
        # 5. 创建PyArrow表并保存
        table = pa.Table.from_arrays([
            pa.array(all_input_ids.reshape(-1, MAX_SEQ_LENGTH).tolist()),
            pa.array(all_attention_mask.reshape(-1, MAX_SEQ_LENGTH).tolist())
        ], names=["input_ids", "attention_mask"])
        
        pq.write_table(table, output_path)
        
        # 6. 清理内存
        del dataset, all_input_ids, all_attention_mask, table
        gc.collect()
        
        elapsed = time.time() - start_time
        print(f"完成处理: {base_name} -> {output_path} "
              f"总耗时: {elapsed:.1f}秒, 平均速度: {total_rows/elapsed:.1f}行/秒")
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        # 清理可能的残留资源
        if 'dataset' in locals():
            del dataset
        gc.collect()

def process_file_wrapper(args):
    """包装函数用于处理内存监控"""
    file_path, idx, total = args
    print(f"\n{'='*50}\n处理文件 [{idx+1}/{total}]: {os.path.basename(file_path)}\n{'='*50}")
    process_file(file_path)
    gc.collect()

if __name__ == "__main__":
    print(f"开始预处理 {len(file_paths)} 个文件，使用 {NUM_PROC} 个进程")
    print(f"每个文件约18万行，总数据量约 {len(file_paths)*180000/1000000:.1f} 百万行")
    
    # 准备参数
    file_args = [(fp, idx, len(file_paths)) for idx, fp in enumerate(file_paths)]
    
    # 使用进程池并行处理
    with mp.Pool(processes=NUM_PROC) as pool:
        results = list(tqdm(pool.imap(process_file_wrapper, file_args), total=len(file_paths)))
    
    print("所有文件处理完成!")
    
    # 生成元数据文件
    metadata = {
        "model": MODEL_PATH,
        "tokenized_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_seq_length": MAX_SEQ_LENGTH,
        "num_files": len(file_paths),
        "output_dir": OUTPUT_DIR
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print(f"元数据已保存到 {OUTPUT_DIR}/metadata.json")