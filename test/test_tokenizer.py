
import json
from transformers import AutoTokenizer

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/han.luzhi/dolly/llm_tokenizer")

# 初始化统计变量
total_characters = 0
total_tokens = 0
num_lines = 0

# 处理单个文本的压缩率计算函数
def calculate_compression(text):
    char_count = len(text)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_count = len(tokens)
    return char_count, token_count

# 处理JSONL文件
with open('/mnt/han.luzhi/dolly/corpus/2020-40_zh_head_0001.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            text = data.get('text', '')  # 获取text字段，若无则返回空字符串
            
            if text:  # 跳过空文本
                chars, tokens = calculate_compression(text)
                total_characters += chars
                total_tokens += tokens
                num_lines += 1
                
        except json.JSONDecodeError:
            print(f"JSON解析错误: {line}")
        except KeyError:
            print(f"行缺少'text'字段: {line}")

# 计算整体压缩率
if total_tokens > 0 and num_lines > 0:
    overall_ratio = total_characters / total_tokens
    avg_chars_per_line = total_characters / num_lines
    avg_tokens_per_line = total_tokens / num_lines
    
    print("\n" + "="*50)
    print(f"文件分析完成: ./corpus/test.jsonl")
    print(f"处理行数: {num_lines}")
    print(f"总字符数: {total_characters}")
    print(f"总Token数: {total_tokens}")
    print("-"*50)
    print(f"整体压缩率: {overall_ratio:.2f} (字符/Token)")
    print(f"平均每行字符数: {avg_chars_per_line:.1f}")
    print(f"平均每行Token数: {avg_tokens_per_line:.1f}")
    print("="*50)
else:
    print("未找到有效文本数据")