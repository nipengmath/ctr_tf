# -*- coding: utf-8 -*-

import json
import codecs
import os
from collections import defaultdict
from datetime import datetime

from sklearn import linear_model, svm
import numpy as np
import tensorflow as tf

from xdeepfm import xDeepFMModel
from util import get_record_parser, get_batch_dataset, get_dataset, cal_metric



def load_data_total(path):
    with codecs.open(path) as f:
        d = json.load(f)
    return d["total"]


def process(config):
    print("Building model...")
    parser = get_record_parser(config)
    test_dataset = get_dataset(config.test_record_file, parser, config)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, test_dataset.output_types, test_dataset.output_shapes)

    test_iterator = test_dataset.make_one_shot_iterator()

    model = xDeepFMModel(config, iterator)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    save_f1 = 0.1
    patience = 0
    lr = config.init_lr

    test_total = load_data_total(config.test_meta)

    with tf.Session(config=sess_config) as sess:
        param_num = sum([np.prod(sess.run(tf.shape(v))) for v in model.all_params])
        print('There are {} parameters in the model'.format(param_num))

        writer = tf.summary.FileWriter(config.event_dir)
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./log/log', sess.graph)
        saver = tf.train.Saver(max_to_keep=10000)

        test_handle = sess.run(test_iterator.string_handle())

        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))

        ## 加载训练好的模型
        if os.path.exists(config.save_dir + "/checkpoint"):
            print("Restoring Variables from Checkpoint.", config.save_dir)
            saver.restore(sess,tf.train.latest_checkpoint(config.save_dir))

        sess.run(tf.assign(model.is_train,
                           tf.constant(False, dtype=tf.bool)))

        test_metrics = evaluate_batch(
            model, test_total, test_total // config.batch_size + 1,
            sess, handle, test_handle)

        print(test_metrics)


def evaluate_batch(model, sample_num, num_batches, sess, handle, str_handle):
    answer_dict = {}
    losses = []
    preds = []
    labels = []
    for _ in range(1, num_batches + 1):
        try:
            print(len(preds))
            step_loss, step_data_loss, \
                step_pred, step_labels, _ = sess.run([model.loss, model.data_loss,
                                                      model.pred, model.labels,
                                                      model.train_op],
                                                     feed_dict={
                                                         handle: str_handle,
                                                         model.layer_keeps: model.keep_prob_test})
        except:
            print("Error: evaluate_batch")
            continue
        losses.append(step_loss)
        preds.extend(step_pred)
        labels.extend(step_labels)

    loss = np.mean(losses)
    preds = preds[:sample_num]
    labels = labels[:sample_num]

    res = cal_metric(labels, preds, model.config)
    # loss_sum = tf.Summary(value=[tf.Summary.Value(
    #     tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    return res


if __name__ == "__main__":
    print("ok")
    from config import flags
    config = flags.FLAGS
    process(config)
