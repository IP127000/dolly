from tokenizers import Tokenizer
from tokenizers.models import BPE  
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))  
# tokenizer.pre_tokenizer = Whitespace()
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
trainer = BpeTrainer(
    vocab_size=12800,           
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],  
    min_frequency=2,            
    show_progress=True         
)

tokenizer.train(files=["corpus/wikipedia.txt"], trainer=trainer)
tokenizer.save("tokenizer_file/tokenizer.json")
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_file/tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
fast_tokenizer.save_pretrained("tokenizer_file/hf_tokenizer_BBPE")