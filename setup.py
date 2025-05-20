from setuptools import setup, find_packages

setup(
    name="llm_dolly",  
    version="0.0.1",      
    packages=find_packages(include=["src_llm"]),  
    install_requires=[
        "torch==2.3.1",
        "datasets==3.6.0",
        "sentencepiece==0.2.0",
        "protobuf==3.20.0",
        "transformers==4.51.3",
        "tokenizers==0.21.1",
        "setuptools==70.3.0",
        "accelerate==1.7.0",
        "tensorboardX==2.6.2.2",
        "deepspeed==0.16.7"
        ],  
    author="IP127000",  
    author_email="hanluzhi@outlook.com", 
    description="LLM-Dolly is a custom LLM built from scratch", 
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown", 
    url="https://github.com/IP127000/LLM-Dolly",
)