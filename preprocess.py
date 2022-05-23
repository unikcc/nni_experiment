#!/usr/bin/env python  
#_*_ coding:utf-8 _*_  

""" 
@author: libobo
@file: preprocess.py 
@time: 20/12/11 0:01
"""
import yaml
import os
import re
import random
import numpy as np
import pickle as pkl
from attrdict import AttrDict
from collections import defaultdict


class Template:
    def __init__(self):
        self.config = AttrDict(yaml.load(open('experiment.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))

    def clean_str_sst(self, string):
        """
        Tokenization/string cleaning for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def read_split_file(self, mode):
        if mode == 'valid': mode = 'dev'
        filename = os.path.join(self.config.data_path, 'stsa.binary.{}'.format(mode))
        a = open(filename, 'r', encoding='utf-8')
        res = []
        for line in a:
            label, text = int(line[0]), self.clean_str_sst(line[1:]).split()
            res.append((text, label))
        return res
        
    def build_dict(self, data_list):
        word_count = defaultdict(int)
        for data in data_list:
            for line, label in data:
                for word in line:
                    word_count[word] += 1
        
        word_count = {k : v for k, v in word_count.items() if v > 5}
        word2dict = {k : i + 1 for i, (k, v) in enumerate(word_count.items())}
        word2dict['UNK'] = 0
        self.word2dict = word2dict

    def execute(self, data_list):
        res = []
        for line, label in data_list:
            res_line = []
            for word in line:
                word = ''.join(['0' if w.isdigit() else w for w in word])
                res_line.append(self.word2dict.get(word, 0))
            res.append((res_line, label))
        return res

    def forward(self):
        modes = ['train', 'valid', 'test']
        data_list = list(map(self.read_split_file, modes))
        self.build_dict(data_list)

        processed_list = list(map(self.execute, data_list))

        if not os.path.exists(self.config['res_path']):
            os.makedirs(self.config['res_path'])
        pkl.dump((processed_list, self.word2dict), open('{}/data.pkl'.format(self.config['res_path']), 'wb'))

if __name__ == '__main__':
    template = Template()
    template.forward()
