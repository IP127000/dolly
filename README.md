# LLM-Dolly
LLM-Dolly is a custom LLM built from scratch   
持续优化更新中...

## 🗓️ 更新日志

#### 2025-05-16
- 📝 添加dolly_llm的预训练代码，进行一次预训练测试：模型0.6B，语料500M，46GB*4显卡.

#### 2025-05-14
- ✅ 将dolly_llm作为pip包，进行安装

#### 2025-05-09
- 📝 优化[RMSNorm](https://arxiv.org/pdf/1910.07467)、MLP、[RoPE](https://arxiv.org/pdf/2104.09864)代码。

#### 2025-05-07
- 📝 使用transformers格式规范modeling和configuration，并设计修改v0.1版modeling_dolly，参数量11.5B。

#### 2025-05-06
- 📝 实现 configuration_dolly类,以及添加v0.0版modeling_dolly。

#### 2025-04-30
- 📝 添加Tokenizer的[BBPE](https://arxiv.org/pdf/1909.03341)方式训练。

#### 2025-04-29
- 📝 添加tokenizer构建代码，支持sentencepiece和transfomers的[BPE](https://arxiv.org/pdf/1508.07909)，支持从文本构建和从已有的tokenzier构建。

#### 2025-04-24
- ✅ 测试从transformers构建自定义的LLM模型结构。

## 致谢

特别感谢以下资源和文章的帮助：

- [Huggingface transformers](https://github.com/huggingface/transformers)
- [Qwen](https://huggingface.co/Qwen)
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909)
- [Neural Machine Translation with Byte-Level Subwords](https://arxiv.org/pdf/1909.03341)
- [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467)
- [RoPE RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)