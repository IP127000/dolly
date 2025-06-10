python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 src_pretrain/pretrain_transformers.py
# torchrun --nproc_per_node=4 src_pretrain/pretrain_transformers.py
# CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 src_pretrain/pretrain_transformers.py
# CUDA_VISIBLE_DEVICES=1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 src_pretrain/pretrain_transformers.py