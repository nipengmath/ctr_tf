# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from prepro import prepro
from train import train

flags = tf.flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

## 临时文件，中间数据
target_dir = "data"
event_dir = "log/event"
save_dir = "log/model"

train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
train_meta = os.path.join(target_dir, "train_meta.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
feature_count_file = os.path.join(target_dir, "feature_count.txt")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(event_dir):
    os.makedirs(event_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

flags.DEFINE_string("mode", "train", "train/debug/test")

flags.DEFINE_string("target_dir", target_dir, "")
flags.DEFINE_string("event_dir", event_dir, "")
flags.DEFINE_string("save_dir", save_dir, "")

flags.DEFINE_string("train_file", "", "")
flags.DEFINE_string("dev_file", "", "")
flags.DEFINE_string("test_file", "", "")

flags.DEFINE_string("train_record_file", train_record_file, "")
flags.DEFINE_string("dev_record_file", dev_record_file, "")
flags.DEFINE_string("test_record_file", test_record_file, "")
flags.DEFINE_string("train_eval_file", train_eval, "")
flags.DEFINE_string("dev_eval_file", dev_eval, "")
flags.DEFINE_string("test_eval_file", test_eval, "")
flags.DEFINE_string("train_meta", train_meta, "")
flags.DEFINE_string("dev_meta", dev_meta, "")
flags.DEFINE_string("test_meta", test_meta, "")
flags.DEFINE_string("feature_count_file", feature_count_file, "")

flags.DEFINE_integer("num_classes", 2, "num of classification classes")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 10, "Number of threads in input pipeline")

## CORPUS
flags.DEFINE_integer("FIELD_COUNT", 66, "")
flags.DEFINE_integer("FEATURE_COUNT", 6656, "")
#flags.DEFINE_integer("buffer_size", 1024*1024*100, "")
flags.DEFINE_integer("buffer_size", 100, "")

## MODEL
flags.DEFINE_string("method", "classification", "")
flags.DEFINE_integer("dim", 10, "Embedding dimension for discrete")
# flags.DEFINE_multi_integer("layer_sizes", [100, 100], "")
flags.DEFINE_multi_integer("layer_sizes", [64, 32], "")
flags.DEFINE_multi_string("activation", ["relu", "relu"], "")
flags.DEFINE_multi_float("dropout", [0.0, 0.0], "")
# flags.DEFINE_multi_integer("cross_layer_sizes", [100, 100, 50], "")
flags.DEFINE_multi_integer("cross_layer_sizes", [64, 32], "")
flags.DEFINE_string("cross_activation", "identity", "")

## TRAIN
flags.DEFINE_integer("epochs", 100, "")
flags.DEFINE_integer("batch_size", 2048, "Batch size")
flags.DEFINE_integer("show_steps", 100, "steps for show")
flags.DEFINE_integer("save_steps", 3000, "steps for save and dev")

flags.DEFINE_string("init_method", "tnormal", "")
flags.DEFINE_float("init_value", 0.01, "")
flags.DEFINE_float("embed_l2", 0.0001, "")
flags.DEFINE_float("embed_l1", 0.0000, "")
flags.DEFINE_float("layer_l2", 0.0001, "")
flags.DEFINE_float("layer_l1", 0.0000, "")
flags.DEFINE_float("cross_l2", 0.0001, "")
flags.DEFINE_float("cross_l1", 0.0000, "")
flags.DEFINE_float("learning_rate", 0.001, "")
flags.DEFINE_string("loss", "log_loss", "")
flags.DEFINE_string("optimizer", "adam", "")

## SHOW
flags.DEFINE_integer("show_step", 1, "")
flags.DEFINE_integer("save_epoch", 5, "")
flags.DEFINE_multi_string("metrics", ["auc", "logloss"], "")

## INFER
flags.DEFINE_string("infer_model_name", "", "")

flags.DEFINE_integer("hidden_size", 50, "hidden size")


flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Num of batches for evaluation")
flags.DEFINE_float("init_lr", 0.0002, "Initial lr for Adadelta")
flags.DEFINE_float("clip_gradients", 3.0, "clip_gradients")
flags.DEFINE_float("keep_prob", 0.7, "Keep prob in rnn")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_integer("patience", 3, "Patience for lr decay")

flags.DEFINE_boolean("is_bucket", False, "Batch size")

tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")


def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "prepro":
        prepro(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()
