# !/usr/bin/env python
# coding: utf-8
# @Author: Dimmy(wangdimmy@gmail.com)
# @Description: Chinese char embedding

import random
from utils.config import *
class MyEmbedding:
    def __init__(self):
        """

        Args:
            show_progress: whether to print progress.

        """
        self.d_emb = 200
        self.model = self.load_word2emb()

    def emb(self, word, default=None):
        if default is None:
            default = self.default
        get_default = {
            'none': lambda: None,
            'zero': lambda: 0.,
            'random': lambda: random.uniform(-0.1, 0.1),
        }[default]
        try:
             g = self.model[word]
             return g
        except KeyError as e:
            return [get_default() for i in range(self.d_emb)]

    def load_word2emb(self, fin_name=args["pretrained_embed_size"]):
        model = {}
        with open(fin_name, "r", encoding="utf-8") as fin:
            next(fin)
            for line in fin:
                elems = line.rstrip().split()
                vec = [float(n) for n in elems[-self.d_emb:]]
                word = elems[0]
                model[word] = vec
        return model

if __name__ == '__main__':
    from time import time
    emb =MyEmbedding(show_progress=True)
    for w in ['canada', 'vancouver', 'toronto']:
        start = time()
        print('embedding {}'.format(w))
        # print(emb.emb(w))
        print('took {}s'.format(time() - start))
