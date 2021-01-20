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


def train(config):
    print("Building model...")
    parser = get_record_parser(config)
    train_dataset = get_dataset(config.train_record_file, parser, config)
    dev_dataset = get_dataset(config.dev_record_file, parser, config)
    test_dataset = get_dataset(config.test_record_file, parser, config)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()
    test_iterator = test_dataset.make_one_shot_iterator()

    model = xDeepFMModel(config, iterator)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    save_f1 = 0.1
    patience = 0
    lr = config.init_lr

    train_total = load_data_total(config.train_meta)
    dev_total = load_data_total(config.dev_meta)
    test_total = load_data_total(config.test_meta)

    with tf.Session(config=sess_config) as sess:
        param_num = sum([np.prod(sess.run(tf.shape(v))) for v in model.all_params])
        for v in model.all_params:
            print(v)

        print('There are {} parameters in the model'.format(param_num))

        writer = tf.summary.FileWriter(config.event_dir)
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./log/log', sess.graph)
        saver = tf.train.Saver(max_to_keep=10000)

        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))

        # ## 加载训练好的模型
        # if os.path.exists(config.save_dir + "/checkpoint"):
        #     print("Restoring Variables from Checkpoint.")
        #     saver.restore(sess,tf.train.latest_checkpoint(config.save_dir))
        #     ## 设置学习率
        #     ## sess.run(tf.assign(model.lr, tf.constant(0.0001, dtype=tf.float32)))

        epoch_steps = int(train_total/config.batch_size)
        num_steps = int(config.epochs * train_total / config.batch_size)
        print("===", config.epochs, train_total, config.batch_size, num_steps)
        for idx in range(1, num_steps + 1):
            ## global_step = sess.run(model.global_step) + 1
            epoch = idx * config.batch_size // train_total
            local_step = idx % epoch_steps
            # print(idx, config.batch_size, train_total, epoch)
            try:
                # step_loss, step_data_loss, _ = sess.run([model.loss, model.data_loss, model.train_op],
                #                                         feed_dict={handle: train_handle,
                #                                                    model.layer_keeps: model.keep_prob_train})

                step_loss, step_data_loss, _ = sess.run([model.loss, model.data_loss, model.update],
                                                        feed_dict={handle: train_handle,
                                                                   model.layer_keeps: model.keep_prob_train})
            except:
                print("Error: this step error. epoch: %s, step: %s" %(epoch, local_step))
                continue

            if idx % config.show_steps == 0:
                print("dt: %s, epoch: %s, local_step: %s, total_step: %s, train loss: %s" %(datetime.now(), epoch, local_step, idx, step_loss))

            # if global_step % config.period == 0:
            #     loss_sum = tf.Summary(value=[tf.Summary.Value(
            #         tag="model/loss", simple_value=loss), ])
            #     writer.add_summary(loss_sum, global_step)

            if idx % config.save_steps == 0:
                sess.run(tf.assign(model.is_train,
                                   tf.constant(False, dtype=tf.bool)))

                dev_metrics = evaluate_batch(
                    model, dev_total, dev_total // config.batch_size + 1,
                    sess, handle, dev_handle)

                test_metrics = evaluate_batch(
                    model, test_total, test_total // config.batch_size + 1,
                    sess, handle, test_handle)

                print("dt: %s, epoch: %s, local_step: %s, total_step: %s, train loss: %s, dev auc: %s, dev logloss: %s, test auc: %s, test logloss: %s" %(datetime.now(), epoch, local_step, idx, step_loss, dev_metrics["auc"], dev_metrics["logloss"], test_metrics["auc"], test_metrics["logloss"]))

                sess.run(tf.assign(model.is_train,
                                   tf.constant(True, dtype=tf.bool)))

                dev_loss = dev_metrics["logloss"]
                if dev_loss < loss_save:
                    loss_save = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= config.patience:
                    lr /= 2.0
                    loss_save = dev_loss
                    patience = 0
                sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                # for s in summ:
                #     writer.add_summary(s, global_step)
                # writer.flush()

                filename = os.path.join(
                    config.save_dir, f"model_{epoch}_{local_step}.ckpt")
                saver.save(sess, filename)
    summary_writer.close()


def evaluate_batch(model, sample_num, num_batches, sess, handle, str_handle):
    answer_dict = {}
    losses = []
    preds = []
    labels = []
    for _ in range(1, num_batches + 1):
        try:
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
