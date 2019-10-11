# !/usr/bin/env python
# coding: utf-8
# @Author: Dimmy(wangdimmy@gmail.com)
# @Description: Train
# @Usage command: python train.py -pes=data/zhwiki.lsn.char -trp=data/train.txt -vap=data/valid.txt -tsp=data/test.txt

from models.csc import *
from utils.prepare_data import *

train, dev, test, lang, confusionset = prepare_data_seq(training=True, batch_size=int(args['batch']))
print("The number of words={}".format(lang.n_words))
model = ConfusionGuide(lang=lang, vocab_size=lang.n_words, embed_size=int(args['embed_size']), hidden_size=int(args['hidden_size']), dropout=args['dropout'], confusionset=confusionset)


bestF1 = 0.0
for epoch in range(200):
    print("Epoch:{}\n".format(epoch))
    # Run the train function
    pbar = tqdm(enumerate(train), total=len(train))
    for i, data in pbar:
        """
        data:
        item_info["src_seqs"] = src_seqs
        item_info["src_len"] = src_lengths
        item_info["tgt_seqs"] = tgt_seqs
        item_info["tgt_gate"] = tgt_gate
       """
        model.train_batch(data["src_seq_index"], data["src_len"], data["tgt_seq_index"], data["tgt_len"], data["tgt_gate"], reset=(i == 0))
        model.optimize(args['clip'])
        pbar.set_description(model.print_loss())

    if (epoch + 1) % int(args['evalp']) == 0:
        print("model evalution")
        bestF1 = model.evaluate(dev, bestF1)



