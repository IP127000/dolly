from src_llm.configuration_dolly import DollyConfig
from transformers.modeling_rope_utils import dynamic_rope_update, ROPE_INIT_FUNCTIONS
import torch

device="cuda"
inv_freq = 1.0 / (10000 ** (torch.arange(0, 128, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / 128))
batch_size = 2
seq_len = 4
position_ids = torch.arange(seq_len).repeat(batch_size, 1).to(device=device)
inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(device)
position_ids_expanded = position_ids[:, None, :].float()
with torch.autocast(device_type="cuda"):  
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() 
    sin = emb.sin() 
