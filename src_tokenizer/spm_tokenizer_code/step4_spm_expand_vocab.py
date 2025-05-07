import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_model_pb2

sp = spm.SentencePieceProcessor(model_file='tokenizer_file/dolly_tokenizer_spm/tokenizer.model')
print(sp.vocab_size())  # 32461

old_model = sp_model_pb2.ModelProto()
old_model.ParseFromString(sp.serialized_model_proto())
pieces = ['拉布拉多', '货拉拉', '拉不拉']
existed_pieces = set(p.piece for p in old_model.pieces)
for piece in pieces:
    if piece not in existed_pieces:
        existed_pieces.add(piece)
        new_piece = sp_model_pb2.ModelProto().SentencePiece()
        new_piece.piece = piece
        new_piece.score = 0
        old_model.pieces.append(new_piece)
    
print(len(old_model.pieces))  # 32461
with open('tokenizer_file/dolly_tokenizer_spm/expand_tokenizer.model', 'wb') as f:
    f.write(old_model.SerializeToString())