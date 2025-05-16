import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM
from transformers import LlamaConfig
config = LlamaConfig(
    vocab_size=32000,  
    hidden_size=4096,  
    num_attention_heads=32,  
    num_hidden_layers=32, 
    intermediate_size=11008,  
)
model = LlamaForCausalLM(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"Model Size: {model_size/1000**2:.1f}M parameters")