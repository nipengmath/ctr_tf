#!/bin/bash
python3 config.py \
    --mode prepro \
    --batch_size 2048 \
    --train_file /mount_nas/newsugg/workspace/nlp/model/search_product_24/train_corpus.txt \
    --dev_file /mount_nas/newsugg/workspace/nlp/model/search_product_24/dev_corpus.txt \
    --test_file /mount_nas/newsugg/workspace/nlp/model/search_product_24/test_corpus.txt
