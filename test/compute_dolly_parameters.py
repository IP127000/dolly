# from transformers import PreTrainedModel
from transformers.modeling_utils import PreTrainedModel
import torch
from torch import nn
from src_llm.configuration_dolly import DollyConfig
from src_llm.modeling_dolly import DollyModel
# model = DollyModel(DollyConfig())
# model.save_pretrained("./dolly_model")
# model.config.save_pretrained("./dolly_model")

config = DollyConfig.from_pretrained("./dolly_model")
model = DollyModel.from_pretrained("./dolly_model", config=config)

# input_ids = torch.tensor([[101, 1024, 102]]).long() 
# outputs = model(input_ids)
# print(outputs)

model_size = sum(t.numel() for t in model.parameters())
print(f"Model Size: {model_size/1000**2:.1f}M parameters")