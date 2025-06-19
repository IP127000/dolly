import os
import multiprocessing as mp
from glob import glob
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = "../corpus"
NUM_PROC = min(32, os.cpu_count())
SAMPLE_SIZE = 10000  #None

tokenizer = AutoTokenizer.from_pretrained("../weights/weights_tokenizer")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_token_lengths(texts):
    return [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]

def process_file(file_path):
    try:
        dataset = load_dataset("parquet", data_files=file_path, split="train")
        if SAMPLE_SIZE and len(dataset) > SAMPLE_SIZE:
            dataset = dataset.shuffle().select(range(SAMPLE_SIZE))
        texts = dataset["text"]
        lengths = get_token_lengths(texts)
        return {
            "file": os.path.basename(file_path),
            "count": len(lengths),
            "mean": np.mean(lengths),
            "std": np.std(lengths),
            "min": np.min(lengths),
            "max": np.max(lengths),
            "percentiles": {
                str(p): np.percentile(lengths, p)
                for p in [50, 75, 90, 95, 99]
            },
            "all_lengths": lengths
        }
    except Exception as e:
        print(f"处理 {file_path} 出错: {e}")
        return None

if __name__ == "__main__":
    file_paths = glob(os.path.join(DATA_DIR, "*.parquet"))
    print(f"找到 {len(file_paths)} 个parquet文件")
    all_stats = []
    with mp.Pool(processes=NUM_PROC) as pool:
        results = pool.imap(process_file, file_paths)
        for stat in tqdm(results, total=len(file_paths)):
            if stat is not None:
                all_stats.append(stat)

    all_lengths = []
    for stat in all_stats:
        all_lengths.extend(stat["all_lengths"])
        stat.pop("all_lengths")

    global_stats = {
        "total_samples": len(all_lengths),
        "global_mean": np.mean(all_lengths),
        "global_max": np.max(all_lengths),
        "global_95_percentile": np.percentile(all_lengths, 95),
        "recommended_max_length": int(np.percentile(all_lengths, 95) * 1.05)
    }

    print(f"\n{'='*50}")
    print("全局统计结果:")
    print(f"总样本数: {global_stats['total_samples']:,}")
    print(f"平均token长度: {global_stats['global_mean']:.1f}")
    print(f"最大token长度: {global_stats['global_max']}")
    print(f"95百分位长度: {global_stats['global_95_percentile']}")
    print(f"建议最大长度: {global_stats['recommended_max_length']}")
