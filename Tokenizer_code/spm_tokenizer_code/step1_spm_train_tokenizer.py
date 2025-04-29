import sentencepiece as spm

###推荐使用 utils/spm_train_tokenizer.sh，使用多核加速

#英文分词器
# corpus_path = 'a.txt,b.txt'
corpus_path_en= 'corpus/pg16457.txt'
vocab_size_en = 1000
model_en= 'tokenizer_file/dolly_tokenizer_spm/tokenizer_en'

# model_type 使用 bpe
model_type = 'bpe'
# byte_fallback=True，避免 out of vocabulary 问题
spm.SentencePieceTrainer.train(input=corpus_path_en, model_prefix=model_en, 
    vocab_size=vocab_size_en, model_type=model_type, byte_fallback=True,character_coverage=1)

#中文分词器
corpus_path_zh= 'corpus/wikipedia.txt'
vocab_size_zh = 32000
model_zh= 'tokenizer_file/dolly_tokenizer_spm/tokenizer_zh'
spm.SentencePieceTrainer.train(input=corpus_path_zh, model_prefix=model_zh, 
    vocab_size=vocab_size_zh, model_type=model_type, byte_fallback=True,character_coverage=1)
