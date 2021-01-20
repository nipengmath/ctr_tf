import tensorflow as tf
import numpy as np
import re
from collections import Counter
import string
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error


def get_record_parser(config, is_test=False):
    """ 解析tfrecord数据格式"""
    def parse(example):
        keys_to_features = {
            'fm_feat_indices': tf.FixedLenFeature([], tf.string),
            'fm_feat_values': tf.VarLenFeature(tf.float32),
            'fm_feat_shape': tf.FixedLenFeature([2], tf.int64),
            'labels': tf.FixedLenFeature([], tf.string),
            'dnn_feat_indices': tf.FixedLenFeature([], tf.string),
            'dnn_feat_values': tf.VarLenFeature(tf.int64),
            'dnn_feat_weights': tf.VarLenFeature(tf.float32),
            'dnn_feat_shape': tf.FixedLenFeature([2], tf.int64),
        }
        parsed = tf.parse_single_example(example, keys_to_features)

        fm_feat_indices = tf.reshape(tf.decode_raw(parsed['fm_feat_indices'], tf.int64), [-1, 2])
        fm_feat_values = tf.sparse_tensor_to_dense(parsed['fm_feat_values'])
        fm_feat_shape = parsed['fm_feat_shape']
        labels = tf.reshape(tf.decode_raw(parsed['labels'], tf.float32), [-1, 1])
        dnn_feat_indices = tf.reshape(tf.decode_raw(parsed['dnn_feat_indices'], tf.int64), [-1, 2])
        dnn_feat_values = tf.sparse_tensor_to_dense(parsed['dnn_feat_values'])
        dnn_feat_weights = tf.sparse_tensor_to_dense(parsed['dnn_feat_weights'])
        dnn_feat_shape = parsed['dnn_feat_shape']
        return fm_feat_indices, fm_feat_values, \
               fm_feat_shape, labels, dnn_feat_indices, \
               dnn_feat_values, dnn_feat_weights, dnn_feat_shape
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id):
            c_len = tf.reduce_sum(
                tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            buckets_min = [np.iinfo(np.int32).min] + buckets
            buckets_max = buckets + [np.iinfo(np.int32).max]
            conditions_c = tf.logical_and(
                tf.less(buckets_min, c_len), tf.less_equal(c_len, buckets_max))
            bucket_id = tf.reduce_min(tf.where(conditions_c))
            return bucket_id

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)

    return dataset


# def get_dataset(record_file, parser, config):
#     num_threads = tf.constant(config.num_threads, dtype=tf.int32)
#     dataset = tf.data.TFRecordDataset(record_file).map(
#         parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
#     return dataset


def get_dataset(record_file, parser, config):
    dataset = tf.data.TFRecordDataset(record_file, buffer_size=config.buffer_size).map(parser).shuffle(config.capacity).prefetch(tf.data.experimental.AUTOTUNE).repeat()

    # buffer_size = tf.constant(config.buffer_size, dtype=tf.int64)
    # num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    # dataset = tf.data.TFRecordDataset(record_file, buffer_size=buffer_size).shuffle(config.capacity).repeat().map(parser, num_parallel_calls=num_threads).prefetch(1)
    return dataset


def cal_metric(labels, preds, config):
    """Calculate metrics,such as auc, logloss, group auc"""
    res = {}
    for metric in config.metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res['auc'] = round(auc, 4)
        if metric == "logloss":
            preds = [max(min(p, 1. - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res['logloss'] = round(logloss, 4)
    return res
