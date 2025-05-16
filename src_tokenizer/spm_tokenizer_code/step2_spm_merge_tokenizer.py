import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_model_pb2

model_file = "tokenizer_part/dolly_tokenizer_spm/tokenizer_zh.model"
en_model_file = "tokenizer_part/dolly_tokenizer_spm/tokenizer_en.model"

model_file_new = "tokenizer_part/dolly_tokenizer_spm/tokenizer.model"

chinese_sp = spm.SentencePieceProcessor(model_file=model_file)
en_sp = spm.SentencePieceProcessor(model_file=en_model_file)

print(chinese_sp.vocab_size())#32000

chinese_spm = sp_model_pb2.ModelProto()
chinese_spm.ParseFromString(chinese_sp.serialized_model_proto())

en_spm = sp_model_pb2.ModelProto()
en_spm.ParseFromString(en_sp.serialized_model_proto())

chinese_sp_set = set(p.piece for p in chinese_spm.pieces)

for p in en_spm.pieces:
    piece = p.piece
    if piece not in chinese_sp_set:
        chinese_sp_set.add(piece)
        new_piece = sp_model_pb2.ModelProto().SentencePiece()
        new_piece.piece = piece
        new_piece.score = 0
        chinese_spm.pieces.append(new_piece)

print(f"after expand vocab={len(chinese_spm.pieces)}") # 32361,增加361个
with open(model_file_new, 'wb') as file:
    file.write(chinese_spm.SerializeToString())