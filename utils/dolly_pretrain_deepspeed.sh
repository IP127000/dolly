deepspeed --num_gpus=4 --master_port=12345 pretrain_transformers.py
# CUDA_VISIBLE_DEVICES=1,2,3 deepspeed --include="localhost:1,2,3" --master_port=12345 src_pretrain/pretrain_deepspeed.py