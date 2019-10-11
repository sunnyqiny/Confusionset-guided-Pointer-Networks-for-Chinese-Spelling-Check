# !/usr/bin/env python
# coding: utf-8
# @Author: Dimmy(wangdimmy@gmail.com)
# @Description: handle data, part of code is referenced from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

import torch.utils.data as torchdata
import torch
import json
from utils.config import *
import pickle
from embed.charzhembed import MyEmbedding
from tqdm import tqdm

class Dataset(torchdata.Dataset):
    def __init__(self, data_info, src_word2id, tgt_word2id):
        """Reads source and target sequences from txt files."""
        self.src_seqs = data_info['src_seq']
        self.tgt_seqs = data_info['tgt_seq']
        self.num_total_seqs = len(self.src_seqs)
        self.src_word2id = src_word2id
        self.tgt_word2id = tgt_word2id

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq_index = self.preprocess(self.src_seqs[index], self.src_word2id)
        tgt_seq_index = self.preprocess(self.tgt_seqs[index], self.tgt_word2id)
        target_gate = [i if src_seq_index[i] == tgt_seq_index[i] else len(src_seq_index)-1 for i, _ in enumerate(tgt_seq_index)]
        target_gate = torch.Tensor(target_gate)

        item_info = {
            "src_seq": "".join(self.src_seqs[index][:-1]),
            "tgt_seq": "".join(self.tgt_seqs[index]),
            "src_seq_index": src_seq_index,
            "tgt_seq_index": tgt_seq_index,
            "tgt_gate": target_gate
        }
        return item_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2idx, is_src=False):
        """Converts words to ids."""
        story = [word2idx[word] if word in word2idx else UNK_token for word in sequence]
        story = torch.Tensor(story)
        return story


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK', SENTINEL_token: "$$$"}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, sent):
        for word in list(sent):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def read_langs(file_name, lang, training):
    print(("Reading from {}".format(file_name)))
    data = []
    with open(file_name,"r", encoding="utf-8") as f:
        for line in f:
            src, tgt = line.strip().split("\t")

            if training:
                lang.index_words(src)
                lang.index_words(tgt)

            data_detail = {
                "src_seq": list(src) + ["$$$"],  # $$$ serves as a sentinel
                "tgt_seq": list(tgt)
            }

            data.append(data_detail)
        print("The size of " + file_name + ":", len(data))
        return data


def get_seq(pairs, lang, batch_size):
    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []

    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k])

    dataset = Dataset(data_info, lang.word2index, lang.word2index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)
    return data_loader


def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(list(seq)) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['src_seq_index']), reverse=True)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['src_seq_index'])
    tgt_seqs, tgt_lengths = merge(item_info["tgt_seq_index"])
    tgt_gate, _ = merge(item_info["tgt_gate"])

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        tgt_seqs = tgt_seqs.cuda()
        tgt_gate = tgt_gate.cuda()

    item_info["src_seq_index"] = src_seqs
    item_info["src_len"] = torch.tensor(src_lengths)
    item_info["tgt_seq_index"] = tgt_seqs
    item_info["tgt_len"] = torch.tensor(tgt_lengths)
    item_info["tgt_gate"] = tgt_gate

    return item_info


def dump_pretrained_emb(word2index, index2word, dump_path):
    print("Dumping pretrained embeddings...")
    embeddings = [MyEmbedding()]
    E = []
    for i in tqdm(range(len(word2index.keys()))):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)

    with open(dump_path, 'wt') as f:
        json.dump(E, f)


def prepare_data_seq(training, batch_size=100):
    print("The batch_size is:", batch_size)
    file_train = args["train_path"]
    file_dev = args["valid_path"]
    file_test = args["test_path"]

    #字典
    lang = Lang()

    #read the confusionset
    confusionset = {}
    with open("data/confusion.txt","r",encoding="utf-8") as file:
        for line in file:
            w, confusion = line.strip().split(":")
            lang.index_words(w+confusion)
            confusionset[w] = list(confusion)

    if training:
        pair_train = read_langs(file_train, lang, training)
        train = get_seq(pair_train, lang, batch_size)
        pair_dev = read_langs(file_dev, lang, training=False)
        dev = get_seq(pair_dev, lang, 1)
        pair_test = read_langs(file_test, lang, training=False)
        test = get_seq(pair_test, lang, 1)

        lang_name = 'lang-all.pkl'

        print("[Info] Loading saved lang files...")
        if not os.path.exists("save/"):
            os.makedirs("save/")

        with open("save/" + lang_name, 'wb') as handle:
            pickle.dump(lang, handle)

        emb_dump_path = 'save/emb{}.json'.format(len(lang.index2word))
        if not os.path.exists(emb_dump_path) and args["load_embedding"]:
            dump_pretrained_emb(lang.word2index, lang.index2word, emb_dump_path)

    return train, dev, test, lang, confusionset


if __name__ == "__main__":
    prepare_data_seq(training=True)









