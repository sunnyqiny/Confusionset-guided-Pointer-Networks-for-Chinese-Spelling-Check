# !/usr/bin/env python
# coding: utf-8
# @Author: Dimmy(wangdimmy@gmail.com)
# @Description: Model

import torch.nn as nn
from utils.config import *
import json
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils.masked_cross_entropy import *
import math
from utils.evaluation_metrics import *

class ConfusionGuide(nn.Module):

    def __init__(self, lang, vocab_size, embed_size, hidden_size, dropout, confusionset, path=None):
        super(ConfusionGuide, self).__init__()
        self.name = "Confusionset-guided model"
        self.lang = lang
        self.confusionset = confusionset
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.Encoder = Encoder(vocab_size, embed_size, hidden_size, dropout, isBidirectional=True)
        self.Decoder = AttnDecoder(lang, vocab_size, embed_size, hidden_size, self.Encoder.Embed, self.Encoder.embed_dropout_layer, self.confusionset)

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                trained_encoder = torch.load(str(path) + '/enc.th')
                trained_decoder = torch.load(str(path) + '/dec.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                trained_encoder = torch.load(str(path) + '/enc.th', lambda storage, loc: storage)
                trained_decoder = torch.load(str(path) + '/dec.th', lambda storage, loc: storage)

            self.Encoder.load_state_dict(trained_encoder.state_dict())
            self.Decoder.load_state_dict(trained_decoder.state_dict())

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        self.reset()

        if USE_CUDA:
            self.Encoder.cuda()
            self.Decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LP:{:.2f}'.format(print_loss_avg, print_loss_ptr)

    def save_model(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.Encoder, directory + '/enc.th')
        torch.save(self.Decoder, directory + '/dec.th')

    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate = 0, 1, 0, 0

    def train_batch(self, input_seqs, input_seqs_len, target_seqs, target_seqs_length, target_gates, reset=0):
        if reset:
            self.reset()

        self.optimizer.zero_grad()
        use_teacher_forcing = random.random() < args["teacher_forcing_ratio"]
        all_vocab_logits, all_gate_logits, all_predict_words = self.encoder_decoder(input_seqs, input_seqs_len, target_seqs, use_teacher_forcing)

        # print("all_vocab_logits:", all_vocab_logits.size())
        # print(all_vocab_logits)
        # print("target_seqs:", target_seqs.size())
        # print(target_seqs)
        # print("target_seqs_length:", target_seqs_length.size())
        # print(target_seqs_length)
        # exit(0)
        loss_vocab = compute_loss(all_vocab_logits, target_seqs, target_seqs_length)
        loss_gate = compute_loss(all_gate_logits, target_gates, target_seqs_length)

        loss = loss_vocab + loss_gate
        self.loss = loss
        self.loss_ptr += loss_vocab.item()
        self.loss_gate += loss_gate.item()

    def optimize(self, clip=10):
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def encoder_decoder(self, input_seqs, encoder_seqs_len, decoder_target_seqs,  use_teacher_forcing=False):
        encoder_outputs, encoder_hidden = self.Encoder(input_seqs.transpose(0, 1), encoder_seqs_len)
        all_vocab_logits, all_gate_logits, all_predict_words = self.Decoder(encoder_outputs, encoder_hidden, input_seqs, encoder_seqs_len, decoder_target_seqs, use_teacher_forcing)

        all_predict_words = np.array(all_predict_words).transpose() # transpose to batch * sequence_length
        return all_vocab_logits, all_gate_logits, all_predict_words

    def evaluate(self, dev, bestF1):
        self.Encoder.train(False)
        self.Decoder.train(False)
        pbar = tqdm(enumerate(dev), total=len(dev))

        final_results = []
        for i, data in pbar:
            """
            data:
            item_info["src_seqs"] = src_seqs
            item_info["src_len"] = src_lengths
            item_info["tgt_seqs"] = tgt_seqs
            item_info["tgt_gate"] = tgt_gate
           """
            all_vocab_logits, all_gate_logits, all_predict_words = self.encoder_decoder(data["src_seq_index"], data["src_len"], data["tgt_seq_index"], False)
            final_results.append((data["src_seq"][0], data["tgt_seq"][0], "".join(all_predict_words[0])))
        detection_f1, correction_f1 = compute_prf(final_results)
        if detection_f1 > bestF1:
            bestF1 = detection_f1
            directory = 'save/models/detectionF1-{}'.format(detection_f1)
            self.save_model(directory)
        return bestF1

class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers = 1, dropout=0.5, isBidirectional=False):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.isBidirectional = isBidirectional
        self.hidden_size = int(hidden_size / 2) if isBidirectional else hidden_size
        self.embed_dropout_layer = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.Embed = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_token)
        self.Embed.weight.data.uniform_(-1, 1)

        if args["load_embedding"]:
            with open(os.path.join("save/", "emb{}.json".format(self.vocab_size))) as file:
                E = json.load(file)
            new = self.Embed.weight.data.new
            self.Embed.weight.data.copy_(new(E))

        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, bidirectional=isBidirectional)
        if USE_CUDA:
            self.lstm = self.lstm.cuda()
            self.embed_dropout_layer = self.embed_dropout_layer()
            self.Embed = self.Embed.cuda()

    def get_state(self, input_seqs):
        batch_size = input_seqs.size(1)
        if self.isBidirectional:
            init_hidden = torch.zeros(2, batch_size, self.hidden_size)
        else:
            init_hidden = torch.zeros(1, batch_size, self.hidden_size)

        if USE_CUDA:
            return (init_hidden.cuda(),init_hidden.cuda())
        else:
            return (init_hidden, init_hidden)

    def forward(self, input_sequence, input_sequence_len=None, hidden=None):
        embed = self.Embed(input_sequence)
        embed = self.embed_dropout_layer(embed)
        hidden = self.get_state(input_sequence)
        if input_sequence_len is not None:
            packpad_embed = nn.utils.rnn.pack_padded_sequence(embed, input_sequence_len, batch_first=False)

        outputs, hidden = self.lstm(packpad_embed, hidden)
        if input_sequence_len is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        if self.isBidirectional:
            hidden = torch.cat((hidden[0][-1],hidden[0][-2]), 1).unsqueeze(0)

        return outputs, hidden


class AttnDecoder(nn.Module):

    def __init__(self, lang, vocab_size, embed_size, hidden_size, shared_embedding_from_encoder,shared_dropoutlayer_from_encoder, confusionset, max_input_len=150):
        super(AttnDecoder, self).__init__()
        self.lang = lang
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.confusionset = confusionset
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embed_dropout_layer = shared_dropoutlayer_from_encoder
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(2 * hidden_size, hidden_size)
        self.gate = nn.Linear(self.hidden_size + max_input_len, max_input_len)
        self.max_input_len = max_input_len
        self.Embed = shared_embedding_from_encoder
        self.sigmoid = nn.Sigmoid()
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.concat = nn.Linear(2 * self.hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, self.vocab_size)
        if USE_CUDA:
            self.U = self.U.cuda()
            self.lstm = self.lstm.cuda()
            self.W1 = self.W1
            self.gate = self.gate.cuda()

    def forward(self, encoder_outputs, encoder_hidden, encoder_seqs_index, encoder_seqs_len, decoder_targets, use_teacher_forcing=False):

        # print("The encoder_outputs size:", encoder_outputs.size())
        # print("The encoder hidden size:", encoder_hidden.size())
        # print("The encoder_seqs_index size:", encoder_seqs_index.size())

        max_decoder_len = max(encoder_seqs_len)-1
        batch_size = encoder_outputs.size(1)
        all_vocab_logits = torch.zeros(batch_size, max_decoder_len, self.vocab_size)
        all_gate_logits = torch.zeros(batch_size,  max_decoder_len, self.max_input_len)
        if USE_CUDA:
            all_vocab_logits = all_vocab_logits.cuda()
            all_gate_logits = all_gate_logits.cuda()

        all_predict_words = []

        # Prepare input and output variables
        decoder_input = torch.LongTensor([SOS_token] * batch_size)
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
        decoder_hidden = encoder_hidden
        for i in range(max_decoder_len):
            decoder_input = self.Embed(decoder_input)
            decoder_input = self.embed_dropout_layer(decoder_input)
            decoder_output, decoder_hidden = self.lstm(decoder_input.unsqueeze(0), (decoder_hidden, decoder_hidden))
            decoder_hidden = decoder_hidden[0]
            C_j = self.attention(encoder_outputs, decoder_hidden, encoder_seqs_len)  # C_j = B * hidden_size

            wordindex_in_encoder = encoder_seqs_index[:, i]
            M = torch.zeros(1)
            M = M.repeat(batch_size, self.lang.n_words)
            for wordindex in range(wordindex_in_encoder.size(0)):
                tmp_word = self.lang.index2word[wordindex_in_encoder[wordindex].item()]
                if tmp_word in self.confusionset:
                    confusion_words = self.confusionset[tmp_word]
                    confusion_words_index = torch.tensor([self.lang.word2index[w] for w in confusion_words])
                    ones = torch.ones(len(confusion_words))
                    tmp_M = torch.ones(self.lang.n_words)
                    tmp_M.scatter_(0, confusion_words_index, ones)
                    M[wordindex] = tmp_M

            M = M.detach()
            if USE_CUDA:
                M = M.cuda()
            p_vocab_logits = self.U(C_j)

            p_vocab_softmax = F.log_softmax( p_vocab_logits, dim=1)
            p_vocab = p_vocab_softmax.mul(M)


            #p_vocab = self.attention_vocab(self.Embed.weight, C_j, M)
            _, topvi = p_vocab.data.topk(1)

            Loc = torch.zeros(batch_size, self.max_input_len)
            Loc[:, i] = 1
            if USE_CUDA:
                Loc = Loc.cuda()

            pointer_dis = F.softmax(self.gate(torch.cat([C_j, Loc], dim=1)), dim=1)
            _, toppi = pointer_dis.data.topk(1)


            all_vocab_logits[:, i, :] = p_vocab
            all_gate_logits[:, i, :] = pointer_dis

            next_in = [encoder_seqs_index[batch_i][toppi[batch_i].item()].item() if (
                        toppi[batch_i].item() < encoder_seqs_len[batch_i] - 1)
                        else topvi[batch_i].item() for batch_i in range(batch_size)]

            decoder_input = torch.LongTensor(next_in)
            batch_words = [self.lang.index2word[tmp_i.item()] for tmp_i in decoder_input]
            all_predict_words.append(batch_words)

            if use_teacher_forcing:
                decoder_input = decoder_targets.transpose(0,1)[i]

            if USE_CUDA:
                decoder_input = decoder_input.cuda()

        return all_vocab_logits, all_gate_logits, all_predict_words

    def attention(self, encoder_outputs, decoder_hidden, encoder_sqs_len):

        H = decoder_hidden.expand_as(encoder_outputs).transpose(0, 1) # decoder_hidden = 1 * B * hidden size
        encoder_outputs = encoder_outputs.transpose(0, 1)
        energy = torch.tanh(self.W1(torch.cat([H, encoder_outputs], dim=2)))
        energy = energy.transpose(2,1)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # batch_size * 1 * hidden_size
        energy = torch.bmm(v, energy)  # [B*1*T]
        energy = energy.squeeze(1)

        max_len = max(encoder_sqs_len)
        for l in encoder_sqs_len:
            if l < max_len:
                energy[:, l:] = -np.inf
        a = F.softmax(energy, dim=1)
        a = a.unsqueeze(1)  # batch_size * 1 * T
        context = a.bmm(encoder_outputs) # batch_size * 1 * hidden_size

        concat_input = torch.cat((decoder_hidden.squeeze(0), context.squeeze(1)), 1)

        C_j = torch.tanh(self.concat(concat_input))
        return C_j

    # def attention_vocab(self, embed, hidden, M):
    #     scores = hidden.matmul(embed.transpose(1, 0))
    #     softmax_scores = F.softmax(scores, dim=1)
    #     softmax_scores = torch.mul(softmax_scores, M)
    #     return softmax_scores


