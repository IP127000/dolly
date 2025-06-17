import pyarrow.parquet as pq
import pyarrow as pa
import json
import os
import glob

input_pattern = "corpus/parquet/train-*-of-00399.parquet"
output_dir = "corpus/jsonl"

os.makedirs(output_dir, exist_ok=True)

parquet_files = glob.glob(input_pattern)

for parquet_file in parquet_files:
    table = pq.read_table(parquet_file)
    base_name = os.path.basename(parquet_file)
    jsonl_file = os.path.join(output_dir, base_name.replace('.parquet', '.jsonl'))
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for batch in table.to_batches():
            for i in range(batch.num_rows):
                row_dict = {col: batch[col][i].as_py() for col in batch.schema.names}
                f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
    
    print(f"已转换 {parquet_file} 为 {jsonl_file}")