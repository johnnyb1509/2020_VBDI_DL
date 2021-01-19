
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime, timedelta

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
from src.inference_summary import *

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

bert_test_dir = './data/bert_test.csv'
raw_test_dir = './data/test.csv'


# Data BERT has 2 columns 'predict', 'gold'
# Data VNDS has 2 columns 'fullText', 'summary'
x_train, y_train = data_reader(bert_train_dir, 'predict', 'gold', MAX_TEXT_LENGTH, MAX_SUMMARY_LENGTH)
x_test, y_test = data_reader(bert_test_dir, 'predict', 'gold', MAX_TEXT_LENGTH, MAX_SUMMARY_LENGTH)


'''Check distribution'''
# words_dist(data_test_, 'cleaned_text', 'cleaned_summary')

y_raw = y_test.copy() # Make a copy to check later by string

# Preparing the Tokenizer
x_train, x_test, x_voc, x_tokenizer = tokenizer(x_train, x_test, MAX_TEXT_LENGTH)

# Define revert filter
reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index

# Check length word count of start token equal to length of the training data
assert y_tokenizer.word_counts['sossonnm'], len(y_train) 

# Delete summary that only has marking

y_train, x_train = del_if_markOnly(y_train, x_train)
y_test, x_test = del_if_markOnly(y_test, x_test)

# Loading model
encoder_model_dir = './models/3_LSTM_encoder_model'
decoder_model_dir = './models/3_LSTM_decoder_model'

encoder_model = tf.keras.models.load_model(encoder_model_dir)
decoder_model = tf.keras.models.load_model(decoder_model_dir)

#%%
print('Decoding...')
start_time = datetime.now()
full_text = []
origin_sum = []
machine_sum = []
for i in range(10000):
    print('{}__________{}__________{}%_________'.format(datetime.now(), i, round(i/10000 *100, 2)))
    text_revert = seq2text(x_test[i], reverse_source_word_index)
    print("Review:",text_revert)
    full_text.append(text_revert)
    print("Original summary:",y_raw[i])
    origin_sum.append(seq2summary(y_test[i], target_word_index, reverse_target_word_index))
    seq_decode = decode_sequence(encoder_model, 
                                               decoder_model,
                                               target_word_index, reverse_target_word_index,
                                               x_test[i].reshape(1,MAX_TEXT_LENGTH),
                                               MAX_SUMMARY_LENGTH, is_Bi = False)
    print("Predicted summary:",seq_decode)
    machine_sum.append(seq_decode)
    print("\n")

result = pd.DataFrame({'fullText': full_text, 
                       'originSum': origin_sum,
                       'machineSum': machine_sum})

result['originSum'] = result['originSum'].apply(lambda x :  x.replace(' eossonnm','').replace('sossonnm ', ''))
# result.to_csv('./data/predict_google_mimic.csv')
print('Scoring')
rouge_score = rouge_scoring(result, 'originSum', 'machineSum')
end_time = datetime.now()
print('start time: ', start_time)
print('end time: ', end_time)
print('time consuming :', end_time-start_time)
K.clear_session()