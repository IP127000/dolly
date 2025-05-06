from transformers import PretrainedConfig
import logging

logger = logging.getLogger(__name__)

class DollyConfig(PretrainedConfig):
    model_type = "dolly"
    keys_to_ignore_at_inference = ["past_key_values"]

    # 添加张量并行配置
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",  # self-attention 的查询投影使用列并行
        "layers.*.self_attn.k_proj": "colwise",  # self-attention 的键投影使用列并行
        "layers.*.self_attn.v_proj": "colwise",  # self-attention 的值投影使用列并行
        "layers.*.self_attn.o_proj": "rowwise",  # self-attention 的输出投影使用行并行
        "layers.*.mlp.gate_proj": "colwise",    # MLP 的门投影使用列并行
        "layers.*.mlp.up_proj": "colwise",      # MLP 的上升投影使用列并行
        "layers.*.mlp.down_proj": "rowwise",    # MLP 的下降投影使用行并行
    }

    # 添加模型并行配置
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),  # 输入 ID 使用 embed_tokens 层生成嵌入
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),  # layers 层的输入输出
        "norm": (["hidden_states"], ["hidden_states"]),  # 标准化层的输入输出
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window  # we check `use_sliding_window` in the modeling code
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["DollyConfig"]
