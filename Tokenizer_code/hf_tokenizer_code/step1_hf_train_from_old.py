from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("text", data_files="corpus/wikipedia.txt", streaming=True)

def batch_iterator(batch_size=1000):
    for batch in dataset["train"].iter(batch_size):
        yield batch["text"]

old_tokenizer = AutoTokenizer.from_pretrained("tokenizer_file/dolly_tokenizer_hf2")
tokenizer = old_tokenizer.train_new_from_iterator(
    batch_iterator(),
    vocab_size=33000,
)

tokenizer.save_pretrained("new_tokenizer")