from datasets import load_dataset
import os
os.environ["HF_DATASETS_CACHE"] = "./data/"
wikipedia = load_dataset("wikipedia", language="zh")
bookcorpus = load_dataset("bookcorpus")
openwebtext = load_dataset("openwebtext")
cc100 = load_dataset("cc100", language="en")

wikipedia_text = wikipedia['train']['text']
bookcorpus_text= bookcorpus["train"]["text"]
openwebtext_text = openwebtext['train']['text']
cc100_text = cc100['train']['text']
combined_text = wikipedia_text + openwebtext_text+ bookcorpus_text
print("finished")
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=30000, min_frequency=2)
tokenizer.train_from_iterator(combined_text, trainer=trainer)
tokenizer.save("path_to_save_your_tokenizer.json")

