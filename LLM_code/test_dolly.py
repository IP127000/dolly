from transformers import PreTrainedModel
import torch
from torch import nn
from configuration_dolly import DollyConfig
from modeling_dolly import DollyModel
model = DollyModel(DollyConfig())
model.save_pretrained("./my_model")
model.config.save_pretrained("./my_model")


# 加载配置和模型
config = MyModelConfig.from_pretrained("./my_model")
model = MyModel.from_pretrained("./my_model", config=config)

# 使用模型进行推理
input_ids = torch.tensor([[101, 1024, 102]]).long()  # 示例输入
outputs = model(input_ids)
print(outputs)
