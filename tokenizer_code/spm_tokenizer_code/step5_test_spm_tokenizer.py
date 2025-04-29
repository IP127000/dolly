import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='tokenizer_part/dolly-tokenizer/tokenizer.model')

sp_en= spm.SentencePieceProcessor(model_file='tokenizer_part/dolly-tokenizer/tokenizer_en.model')
sentence = 'A Frenchman from Paris'
pieces = sp.encode_as_pieces(sentence)
pieces_en = sp_en.encode_as_pieces(sentence)
print(pieces)  # ['▁A', '▁F', 'ren', 'ch', 'man', '▁from', '▁P', 'ar', 'is']
print(pieces_en) #'▁A', '▁F', 'ren', 'ch', 'man', '▁from', '▁P', 'aris'
tokens = sp.encode(sentence)
tokens_en = sp_en.encode(sentence)
print(tokens)  # [338, 493, 587, 349, 807, 429, 373, 279, 282]
print(tokens_en) #364, 490, 5981, 398, 2036, 32046, 386, 9020] 
# 解码后的结果应等于编码前的输入
# assert sp.decode(tokens) == sentence