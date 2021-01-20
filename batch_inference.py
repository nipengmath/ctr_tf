# -*- coding: utf-8 -*-

import os, sys, codecs, re
import tensorflow as tf
import numpy as np
import json
import jieba
from util import to_unicode
# import ujson as json

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from func import cudnn_gru, native_gru, dot_attention, summ, ptr_net, dense, dropout, bi_shortcut_stacked_lstm_return_sequences

from prepro import word_tokenize

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# # Must be consistant with training
para_max_num = 5
para_max_length = 10
hidden_size = 50

num_classes = 3

decay_steps = 1000
decay_rate = 1.0
clip_gradients = 3.0

# File path
target_dir = "data"
save_dir = "log/model"
word_emb_file = os.path.join(target_dir, "word_emb.json")
word2idx_file = os.path.join(target_dir, "word2idx.json")


class InfModel(object):

    def __init__(self, word_mat, batch_size):
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.c_maxnum = para_max_num
        self.c_maxlen = para_max_length

        self.c = tf.placeholder(tf.int32, [batch_size, para_max_num, para_max_length])
        self.keep_prob = 1.0
        self.is_train = None
        self.batch_size = batch_size

        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)

        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.clip_gradients = clip_gradients

        self.ready()

    def ready(self):
        N, PN, PL, HS = self.batch_size, self.c_maxnum, self.c_maxlen, self.hidden_size
        gru = native_gru

        self.c_mask = tf.cast(tf.reshape(self.c, [N*PN, PL]), tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)

        with tf.variable_scope("emb"):
            c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
            c_emb = tf.reshape(c_emb, [N*PN, PL, c_emb.get_shape().as_list()[-1]])

        with tf.variable_scope("word_encoder"):
            rnn = gru(num_layers=1, num_units=HS, batch_size=N*PN, input_size=c_emb.get_shape(
            ).as_list()[-1], keep_prob=self.keep_prob, is_train=self.is_train, scope="word")
            c_encoder = rnn(c_emb, seq_len=self.c_len)

        with tf.variable_scope("word_attention"):
            dim = c_encoder.get_shape().as_list()[-1]
            u = tf.nn.tanh(dense(c_encoder, dim, use_bias=True, scope="dense"))
            u2 = tf.reshape(u, [N*PN*PL, dim])
            uw = tf.get_variable("uw", [dim, 1])
            alpha = tf.matmul(u2, uw)
            alpha = tf.reshape(alpha, [N*PN, PL])
            alpha = tf.nn.softmax(alpha, axis=1)
            s = tf.matmul(tf.expand_dims(alpha, axis=1), c_encoder)

        with tf.variable_scope("sent_encoder"):
            dim = s.get_shape().as_list()[-1]
            s = tf.reshape(s, [N, PN, dim])
            s_len = tf.constant([PN for _ in range(N)],shape=[N,], name='s_len')
            rnn = gru(num_layers=1, num_units=HS, batch_size=N, input_size=dim,
                      keep_prob=self.keep_prob, is_train=self.is_train, scope="sent")
            h = rnn(s, seq_len=s_len)

        with tf.variable_scope("sent_attention"):
            dim = s.get_shape().as_list()[-1]
            u = tf.nn.tanh(dense(s, dim, use_bias=True, scope="dense"))
            u2 = tf.reshape(u, [N*PN, dim])
            us = tf.get_variable("us", [dim, 1])
            alpha2 = tf.matmul(u2, us)
            alpha2 = tf.reshape(alpha2, [N, PN])
            alpha2 = tf.nn.softmax(alpha2, axis=1)
            v = tf.matmul(tf.expand_dims(alpha2, axis=1), s)
            v = tf.reshape(v, [N, dim])

        with tf.variable_scope("output"):
            self.logits = dense(v, self.num_classes,
                                use_bias=True, scope="output")
            self.scores = tf.nn.softmax(self.logits)

        #with tf.variable_scope("predict"):
        #    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        #        logits=self.logits,
        #        labels=tf.stop_gradient(self.y))
        #    self.loss = tf.reduce_mean(losses)


class Inference(object):

    def __init__(self, batch_size):
        with open(word_emb_file, "r") as fh:
            self.word_mat = np.array(json.load(fh), dtype=np.float32)
        with open(word2idx_file, "r") as fh:
            self.word2idx_dict = json.load(fh)
        self.model = InfModel(self.word_mat, batch_size)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(save_dir))

        self.batch_size = batch_size
        self.para_max_num = para_max_num
        self.para_max_length = para_max_length

    def _response(self, contexts):
        sess = self.sess
        model = self.model
        context_idxs = self.prepro(contexts)
        yp = sess.run(model.scores,
                      feed_dict={model.c: context_idxs})
        return yp

    def prepro(self, contexts):
        num = len(contexts)

        context_tokens = []
        for text in contexts:
            text = to_unicode(text)
            lst = re.split(r",|\?|!|。|，|？|！", text)
            tokens = []
            for x in lst:
                para_tokens = word_tokenize(x)
                tokens.append(para_tokens)
            context_tokens.append(tokens)

        context_idxs = np.zeros([self.batch_size, self.para_max_num, self.para_max_length], dtype=np.int32)

        def _get_word(each):
            if each in self.word2idx_dict:
                return self.word2idx_dict[each]
            return 1

        for b, context_token in enumerate(context_tokens):
            for i, tokens in enumerate(context_token):
                if i < para_max_num:
                    for j, token in enumerate(tokens):
                        if j < para_max_length:
                            context_idxs[b, i,j] = _get_word(token)
        return context_idxs

    def response(self, contexts):
        rst = []
        start = 0
        while True:
            end = start + self.batch_size
            print(start, end)
            _contexts = contexts[start: end]

            n_sample = len(_contexts)
            if n_sample == 0:
                break
            start = end
            if n_sample < self.batch_size:
                _contexts.extend(["" for _ in range(self.batch_size-n_sample)])

            _yp = self._response(_contexts)
            _yp = list(_yp[:n_sample, :])
            rst.extend(_yp)
        return rst


def test():
    batch_size = 10
    infer = Inference(batch_size)
    contexts = [
        "凡普金科响应监管 完善管控机制推动普惠金融发展",
        "联璧金融遭挤兑 唐小僧等高返四天王疑全线阵亡",
        "人人贷理财入选菲特财经网和网贷之家5月P2P网贷百强榜"
    ]
    scores = infer.response(contexts)
    print(scores)


if __name__ == "__main__":
    test()
