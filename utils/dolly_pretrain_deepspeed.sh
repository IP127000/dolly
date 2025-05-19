deepspeed --num_gpus=4 --master_port=12345 pretrain_transformers.py
#   deepspeed --include="localhost:1,2,3" --master_port=12345 pretrain_transformers.py