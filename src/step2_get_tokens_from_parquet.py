import os
import multiprocessing as mp
from glob import glob
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import gc
import psutil
import time

DATA_DIR = "../corpus"
OUTPUT_DIR = "../corpus/tokens"  
MAX_SEQ_LENGTH = 1024       
NUM_PROC = min(32, psutil.cpu_count(logical=False))  
CHUNK_SIZE = 50000           
MAX_MEMORY_PERCENT = 70   
os.makedirs(OUTPUT_DIR, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained("../weights/weights_tokenizer")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

file_paths =  glob(os.path.join(DATA_DIR, "*.parquet"))
print(f"找到 {len(file_paths)} 个parquet文件")

def memory_safe():
    mem = psutil.virtual_memory()
    return mem.percent < MAX_MEMORY_PERCENT

def process_chunk(texts, tokenizer):
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors="np"
    )
    return tokenized["input_ids"], tokenized["attention_mask"]

def process_file(file_path):
    try:
        base_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_DIR, f"tokenized_{base_name}")
        if os.path.exists(output_path):
            print(f"文件已存在: {output_path}, 跳过")
            return
        print(f"开始处理: {base_name}")
        start_time = time.time()
        dataset = load_dataset("parquet", data_files=file_path, split="train")
        total_rows = len(dataset)
        print(f"文件包含 {total_rows:,} 行")
        all_input_ids = []
        all_attention_mask = []
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
            chunk = dataset.select(range(start_row, end_row))
            texts = chunk["text"] 
            input_ids, attention_mask = process_chunk(texts, tokenizer)
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            if (chunk_idx + 1) % 5 == 0 or (chunk_idx + 1) == num_chunks:
                elapsed = time.time() - start_time
                rows_done = min((chunk_idx + 1) * CHUNK_SIZE, total_rows)
                percent = rows_done / total_rows * 100
                print(f"进度: {rows_done:,}/{total_rows:,} 行 ({percent:.1f}%) "
                      f"耗时: {elapsed:.1f}秒")
        all_input_ids = np.vstack(all_input_ids)
        all_attention_mask = np.vstack(all_attention_mask)
        table = pa.Table.from_arrays([
            pa.array(all_input_ids.reshape(-1, MAX_SEQ_LENGTH).tolist()),
            pa.array(all_attention_mask.reshape(-1, MAX_SEQ_LENGTH).tolist())
        ], names=["input_ids", "attention_mask"])
        pq.write_table(table, output_path)
        del dataset, all_input_ids, all_attention_mask, table
        gc.collect()
        elapsed = time.time() - start_time
        print(f"完成处理: {base_name} -> {output_path} "
              f"总耗时: {elapsed:.1f}秒, 平均速度: {total_rows/elapsed:.1f}行/秒")
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        if 'dataset' in locals():
            del dataset
        gc.collect()

def process_file_wrapper(args):
    file_path, idx, total = args
    print(f"\n{'='*50}\n处理文件 [{idx+1}/{total}]: {os.path.basename(file_path)}\n{'='*50}")
    process_file(file_path)
    gc.collect()

if __name__ == "__main__":
    print(f"开始预处理 {len(file_paths)} 个文件，使用 {NUM_PROC} 个进程")
    file_args = [(fp, idx, len(file_paths)) for idx, fp in enumerate(file_paths)]
    with mp.Pool(processes=NUM_PROC) as pool:
        results = list(tqdm(pool.imap(process_file_wrapper, file_args), total=len(file_paths)))
    print("所有文件处理完成!")