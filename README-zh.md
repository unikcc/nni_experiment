## NNI教程

[English](./README.md) | 简体中文

+ 本项目的基础代码是基于PyTorch实现的TextCNN模型，用于文本分类任务

+ 安装依赖
```pip install requirements.txt```

+ 你可以通过以下代码直接执行原始的文本分类任务：
```bash
python preprocess.py
python main.py
```
+ 运行以下命令可以实现使用NNI自动调节参数：
```bash
sh nni.sh
```
+ 本项目的详细信息在博客中有介绍，博客[https://www.jianshu.com/p/d88643a91e7d](https://www.jianshu.com/p/d88643a91e7d)