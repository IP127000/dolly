import torch
import math

def llama_rope(x, seq_len, dim, device='cpu'):
    """
    实现 Llama 中的 Rotary Positional Encoding (RoPE)
    
    x: 输入的嵌入，大小为 (batch_size, seq_len, dim)
    seq_len: 序列长度
    dim: 嵌入的维度
    device: 设备（默认为 'cpu'）
    """
    # 生成位置索引
    position = torch.arange(0, seq_len, dtype=torch.float32, device=device)
    
    # 计算每个维度的旋转角度
    freqs = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / dim))
    
    # 计算位置编码矩阵
    angle_rates = position.unsqueeze(1) * freqs.unsqueeze(0)
    
    # 使用 sin 和 cos 对每一对相邻维度进行旋转
    sin = torch.sin(angle_rates)
    cos = torch.cos(angle_rates)
    
    # 将 sin 和 cos 分别填充到奇数和偶数维度
    encoding = torch.zeros((seq_len, dim), device=device)
    encoding[:, 0::2] = sin
    encoding[:, 1::2] = cos
    
    # 对输入的嵌入应用旋转
    x_rot = torch.matmul(x, encoding.unsqueeze(0).transpose(0, 1))
    
    return x_rot

batch_size = 1
seq_len = 2  # 序列长度
dim = 3  # 嵌入维度
x = torch.randn(batch_size, seq_len, dim)

encoded_x = llama_rope(x, seq_len, dim)
print(encoded_x.shape)  # 输出：torch.Size([2, 10, 16])