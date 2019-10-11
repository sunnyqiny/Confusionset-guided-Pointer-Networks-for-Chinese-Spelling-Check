# !/usr/bin/env python
# coding: utf-8
# @Author: Dimmy(wangdimmy@gmail.com)
# @Description: Initialize some parameters needed to be used in our model

import os
import argparse
if os.cpu_count() > 8:
    USE_CUDA = True
else:
    USE_CUDA = False

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0
SENTINEL_token = 4

parser = argparse.ArgumentParser("Parameters for the model")
parser.add_argument("-le","--load_embedding",help="Loading a pretrained embedding", required=False, default=1, type=int)
parser.add_argument('-bsz','--batch', help='Batch_size', required=False, type=int, default=4)
parser.add_argument('-hdsize','--hidden_size', help='Hidden size', required=False, type=int, default=512)
parser.add_argument('-embsize','--embed_size', help='Embedding size', required=False, type=int, default=200)

parser.add_argument('-pes','--pretrained_embed_size', help='Pretained embedding path', required=False, type=str)

parser.add_argument('-trp','--train_path', help='Training data path, exmaple: "右手司机自己轿车。\t右手司机自己叫车。" for each line', required=True, type=str)
parser.add_argument('-vap','--valid_path', help='Validating data path, exmaple: "右手司机自己轿车。\t右手司机自己叫车。" for each line', required=True, type=str)
parser.add_argument('-tsp','--test_path',  help='Testing data path, exmaple: "右手司机自己轿车。\t右手司机自己叫车。" for each line', required=True, type=str)

parser.add_argument('-clip','--clip', help='gradient clipping', required=False, default=10, type=int)
parser.add_argument('-all_vocab','--all_vocab', help='', required=False, default=1, type=int)
parser.add_argument('-tfr','--teacher_forcing_ratio', help='teacher_forcing_ratio',
                    type=float, required=False, default=0.5)
parser.add_argument('-dr','--dropout', help='Drop Out', required=False, type=float, default=0.5)

parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=1)
args = vars(parser.parse_args())

print(args)


