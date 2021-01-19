# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:56:34 2021

@author: NguyenSon
"""

import numpy as np
import pandas as pd
import os
import re

from keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Concatenate, \
                                    Embedding, Dense, Concatenate
                                    
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

from src.utils_ import *
from src.base_model import *
from src.attention import AttentionLayer

import warnings
warnings.filterwarnings("ignore")

'''GPU Growth = True'''
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


MAX_TEXT_LENGTH = 200
MAX_SUMMARY_LENGTH = 14

latent_dim = 340
embedding_dim = 100

bert_train_dir = './data/bert_train.csv'
raw_train_dir = './data/train.csv'

bert_valid_dir = './data/bert_test.csv'
raw_valid_dir = './data/valid.csv'

# Data BERT has 2 columns 'predict', 'gold'
# Data VNDS has 2 columns 'fullText', 'summary'

x_train, y_train = data_reader(bert_train_dir, 'predict', 'gold', MAX_TEXT_LENGTH, MAX_SUMMARY_LENGTH)
x_valid, y_valid = data_reader(bert_valid_dir, 'predict', 'gold', MAX_TEXT_LENGTH, MAX_SUMMARY_LENGTH)


'''Check distribution'''
# words_dist(data_train_, 'cleaned_text', 'cleaned_summary')
# words_dist(data_valid_, 'cleaned_text', 'cleaned_summary')

y_raw = y_train.copy() # Make a copy to check later by string

# Preparing the Tokenizer

x_train, x_valid, x_voc, x_tokenizer = tokenizer(x_train, x_valid, MAX_TEXT_LENGTH)
y_train, y_valid, y_voc, y_tokenizer = tokenizer(y_train, y_valid, MAX_TEXT_LENGTH)

# Define revert filter
reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index

# Check length word count of start token equal to length of the training data
assert y_tokenizer.word_counts['sossonnm'], len(y_train) 

# Delete summary that only has marking

y_train, x_train = del_if_markOnly(y_train, x_train)

y_validate, x_validate = del_if_markOnly(y_validate, x_validate)


# Model building

full_model, encoder_model, decoder_model = base_model_LSTM(x_voc, y_voc, 
                                                            latent_dim, 
                                                            embedding_dim, 
                                                            MAX_TEXT_LENGTH,
                                                            is_save = False)

full_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)

history = full_model.fit([x_train,y_train[:,:-1]], 
                         y_train.reshape(y_train.shape[0],
                                         y_train.shape[1], 1)[:,1:],
                         epochs=10,
                         callbacks=[es],
                         batch_size=36, 
                         validation_data=(
                                          [x_valid,y_valid[:,:-1]], 
                                           y_valid.reshape(y_valid.shape[0],
                                                          y_valid.shape[1], 1)[:,1:]
                                           )
                         )

# Loss curve plot
loss_curve(history)
K.clear_session()

