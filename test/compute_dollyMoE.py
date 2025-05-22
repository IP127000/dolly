from transformers.modeling_utils import PreTrainedModel
import torch
from torch import nn
from src_llm_moe.configuration_dollyMoE import DollyMoEConfig
from src_llm_moe.modeling_dollyMoE import DollyMoEModel,DollyMoEForCausalLM
model = DollyMoEForCausalLM(DollyMoEConfig())
model.save_pretrained("./weights_MoE")
model.config.save_pretrained("./weights_MoE")

config = DollyMoEConfig.from_pretrained("./weights_MoE")
model = DollyMoEForCausalLM.from_pretrained("./weights_MoE", config=config)

input_ids = torch.tensor([[101, 1024, 102]]).long() 
outputs = model(input_ids)
print(outputs)

model_size = sum(t.numel() for t in model.parameters())
print(f"Model Size: {model_size/1000**2:.1f}M parameters")