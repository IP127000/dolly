from configuration_dolly import DollyConfig
from modeling_dolly import DollyModel

model = DollyModel(DollyConfig())

#保存配置和权重
model.config.save_pretrained("../weights/weights_llm")
model.save_pretrained("../weights/weights_llm")

#加载配置和权重
config = DollyConfig.from_pretrained("../weights/weights_llm")
model = DollyModel.from_pretrained("../weights/weights_llm", config=config)

#计算参数量
model_size = sum(t.numel() for t in model.parameters())
print(f"Model Size: {model_size/1000**2:.1f}M parameters")

#测试模型
# input_ids = torch.tensor([[101, 1024, 102]]).long() 
# outputs = model(input_ids)
# print(outputs)