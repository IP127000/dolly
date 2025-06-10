from transformers import AutoTokenizer

def get_training_corpus(file_path, batch_size=1000):
    with open(file_path, "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            batch.append(line.strip())  
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  
            yield batch

old_tokenizer = AutoTokenizer.from_pretrained("tokenizer_file/dolly_tokenizer_hf2")  
training_corpus = get_training_corpus("corpus/wikipedia.txt")  

tokenizer = old_tokenizer.train_new_from_iterator(
    training_corpus,
    vocab_size=34000
)


tokenizer.save_pretrained("tokenizer_file/new_tokenizer2")