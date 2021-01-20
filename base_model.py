# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net, dense, bi_shortcut_stacked_lstm_return_sequences


class BaseModel(object):
    def __init__(self, config, batch, trainable=True, opt=True):
        self.config = config

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        self.fm_feat_indices, self.fm_feat_values, \
            self.fm_feat_shape, self.labels, \
            self.dnn_feat_indices, self.dnn_feat_values, \
            self.dnn_feat_weights, self.dnn_feat_shape = batch.get_next()

        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)

        self.num_classes = config.num_classes

        self.decay_steps, self.decay_rate = config.decay_steps, config.decay_rate
        self.clip_gradients = config.clip_gradients
        self.hidden_size = config.hidden_size
        self.keep_prob = config.keep_prob

        self.embed_params = []
        self.layer_params = []
        self.cross_params = []

        ## 初始化
        self.initializer = self._get_initializer()

        ## 构建网络结构
        self.logit = self._build_graph()

        ##
        self.pred = self._get_pred(self.logit)
        self.data_loss = self._compute_data_loss()
        self.regular_loss = self._compute_regular_loss()
        self.loss = tf.add(self.data_loss, self.regular_loss)
        self.saver = tf.train.Saver(max_to_keep=self.config.epochs)
        self.update = self._build_train_opt()
        self.init_op = tf.global_variables_initializer()
        self.merged = self._add_summaries()

        self.all_params = tf.trainable_variables()

        if trainable:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def _get_pred(self, logit):
        if self.config.method == 'regression':
            pred = tf.identity(logit)
        elif self.config.method == 'classification':
            pred = tf.sigmoid(logit)
        else:
            raise ValueError("method must be regression or classification, but now is {0}".format(self.config.method))
        return pred

    def _add_summaries(self):
        tf.summary.scalar("data_loss", self.data_loss)
        tf.summary.scalar("regular_loss", self.regular_loss)
        tf.summary.scalar("loss", self.loss)
        merged = tf.summary.merge_all()
        return merged

    def _l2_loss(self):
        l2_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l2_loss = tf.add(l2_loss, tf.multiply(self.config.embed_l2, tf.nn.l2_loss(param)))
        params = self.layer_params
        for param in params:
            l2_loss = tf.add(l2_loss, tf.multiply(self.config.layer_l2, tf.nn.l2_loss(param)))
        return l2_loss

    def _l1_loss(self):
        l1_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l1_loss = tf.add(l1_loss, tf.multiply(self.config.embed_l1, tf.norm(param, ord=1)))
        params = self.layer_params
        for param in params:
            l1_loss = tf.add(l1_loss, tf.multiply(self.config.layer_l1, tf.norm(param, ord=1)))
        return l1_loss

    def _cross_l_loss(self):
        cross_l_loss = tf.zeros([1], dtype=tf.float32)
        for param in self.cross_params:
            cross_l_loss = tf.add(cross_l_loss, tf.multiply(self.config.cross_l1, tf.norm(param, ord=1)))
            cross_l_loss = tf.add(cross_l_loss, tf.multiply(self.config.cross_l2, tf.norm(param, ord=1)))
        return cross_l_loss

    def _get_initializer(self):
        if self.config.init_method == 'tnormal':
            return tf.truncated_normal_initializer(stddev=self.config.init_value)
        elif self.config.init_method == 'normal':
            return tf.random_normal_initializer(stddev=self.config.init_value)
        else:
            return tf.truncated_normal_initializer(stddev=self.config.init_value)

    def _compute_data_loss(self):
        if self.config.loss == 'cross_entropy_loss':
            data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=tf.reshape(self.logit, [-1]),
                labels=tf.reshape(self.labels, [-1])))
        elif self.config.loss == 'square_loss':
            data_loss = tf.sqrt(tf.reduce_mean(
                tf.squared_difference(tf.reshape(self.pred, [-1]),
                                      tf.reshape(self.labels, [-1]))))
        elif self.config.loss == 'log_loss':
            data_loss = tf.reduce_mean(tf.losses.log_loss(
                predictions=tf.reshape(self.pred, [-1]),
                labels=tf.reshape(self.labels, [-1])))
        else:
            raise ValueError("this loss not defined {0}".format(self.config.loss))
        return data_loss

    def _compute_regular_loss(self):
        regular_loss = self._l2_loss() + self._l1_loss() + self._cross_l_loss()
        regular_loss = tf.reduce_sum(regular_loss)
        return regular_loss

    def _build_train_opt(self):
        if self.config.optimizer == 'adam':
            train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)
        else:
            train_step = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss)
        return train_step

    def _active_layer(self, logit, scope, activation, layer_idx):
        logit = self._dropout(logit, layer_idx)
        logit = self._activate(logit, activation)
        return logit

    def _activate(self, logit, activation):
        if activation == 'sigmoid':
            return tf.nn.sigmoid(logit)
        elif activation == 'softmax':
            return tf.nn.softmax(logit)
        elif activation == 'relu':
            return tf.nn.relu(logit)
        elif activation == 'tanh':
            return tf.nn.tanh(logit)
        elif activation == 'elu':
            return tf.nn.elu(logit)
        elif activation == 'identity':
            return tf.identity(logit)
        else:
            raise ValueError("this activations not defined {0}".format(activation))

    def _dropout(self, logit, layer_idx):
        logit = tf.nn.dropout(x=logit, keep_prob=self.layer_keeps[layer_idx])
        return logit

    def train(self, sess, handle):
        return sess.run([self.update, self.loss, self.data_loss, self.merged], \
                        feed_dict={self.layer_keeps: self.keep_prob_train,
                                   self.handle: handle})

    def eval(self, sess, handle):
        return sess.run([self.loss, self.data_loss, self.pred, self.labels], \
                        feed_dict={self.layer_keeps: self.keep_prob_test, handle: handle})

    def infer(self, sess, handle):
        return sess.run([self.pred], \
                        feed_dict={self.layer_keeps: self.keep_prob_test, handle: handle})

    def build_graph(self):
        pass
