# LLM-Dolly
LLM-Dolly is a custom LLM built from scratch   
持续优化更新中...

## 🗓️ 更新日志

#### 2025-05-09
- 📝 优化[RMSNorm](https://blog.csdn.net/m0_57102661/article/details/142741221)、MLP、[RoPE](https://zhuanlan.zhihu.com/p/359502624)代码。

#### 2025-05-07
- 📝 使用transformers格式规范modeling和configuration，并设计修改v0.1版modeling_dolly，参数量11.5B。

#### 2025-05-06
- 📝 实现 configuration_dolly类,以及添加v0.0版modeling_dolly。

#### 2025-04-30
- 📝 添加Tokenizer的[BBPE](https://zhuanlan.zhihu.com/p/3329211354)方式训练。

#### 2025-04-29
- 📝 添加tokenizer构建代码，支持sentencepiece和transfomers方式，支持从文本构建和从已有的tokenzier构建。

#### 2025-04-24
- ✅ 测试从transformers构建自定义的LLM模型结构。

## 致谢

特别感谢以下资源和文章的帮助：

- [HelloWorld：tokenizer](https://zhuanlan.zhihu.com/p/657047389)
- [NLP从0到1之HuggingFace实战](https://zhuanlan.zhihu.com/p/690019010)
- [Huggingface transformers](https://github.com/huggingface/transformers)
- [Qwen](https://huggingface.co/Qwen)