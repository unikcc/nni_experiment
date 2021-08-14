## NNI Tutorial

English | [简体中文](./README-zh.md)


The basic code of this project is implemented with Pytorch for sentiment classification task.
You can execute the original task via:
```bash
python preprocess.py
python main.py
```
We use NNI [link](https://github.com/microsoft/nni) in this project to auto fine tune model parameter.
To run the code with nni:
```bash
sh nni.sh
```
The blog(Chinese version) related to this project is at [https://www.jianshu.com/p/d88643a91e7d](https://www.jianshu.com/p/d88643a91e7d).