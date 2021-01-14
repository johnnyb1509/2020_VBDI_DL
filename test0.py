# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:56:34 2021

@author: NguyenSon
"""
from attention_keras.src.layers.attention import AttentionLayer
import numpy as np
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

from keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Concatenate, \
                                    Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
#%%
from src.utils_dl_proj.utils_ import *
# import warnings
# warnings.filterwarnings("ignore")

#%%
# '''GPU Growth = True'''
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)

#%%
data_train = pd.read_csv('./data/train.csv', usecols=['summary', 'fullText'])
data_test = pd.read_csv('./data/test.csv', usecols=['summary', 'fullText'])
data_val = pd.read_csv('./data/valid.csv', usecols=['summary', 'fullText'])
#%%
data_train = del_short_summary(data_train, 'summary')
data_test = del_short_summary(data_test, 'summary')
data_val = del_short_summary(data_val, 'summary')

data_train = pd.concat([data_train, data_val])
#%%
# Text cleaning both full text and summary
data_train_ = frame_clean(data_train, 'fullText', 'summary')
data_test_ = frame_clean(data_test, 'fullText', 'summary')
#%%
'''Check distribution'''
# words_dist(data_train_, 'cleaned_text', 'cleaned_summary')
# words_dist(data_test_, 'cleaned_text', 'cleaned_summary')

#%%
# Define maximum text length and summary length for training
MAX_TEXT_LENGTH = 200
MAX_SUMMARY_LENGTH = 14

data_ = filter_data_length(data_train_, 'cleaned_text', 'cleaned_summary', 
                                MAX_TEXT_LENGTH, MAX_SUMMARY_LENGTH)

# Marking sos and eos for summary sentence
data_= mark_sentence(data_, 'cleaned_summary')
#%%
df = data_.copy()
x_tr,x_val,y_tr,y_val=train_test_split(np.array(df['cleaned_text']),
                                       np.array(df['cleaned_summary']),
                                       test_size=0.2,
                                       random_state=42,
                                       shuffle=True)
#%%
# '''define value'''
# x_tr = data_train['fullText'].values
# x_val = data_val['fullText'].values

# y_tr = data_train['fullText'].values
# y_val = data_val['fullText'].values

#%%
# # Preparing the Tokenizer
# 
# A tokenizer builds the vocabulary and converts a word sequence to an integer sequence. Go ahead and build tokenizers for text and summary:
# 
# # Text Tokenizer

token_filter = '! "# $% & () * +, -. / : ; <=>? @ [] ^` {|} ~ “ ” ‘ ’'
#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer(filters = token_filter) 
x_tokenizer.fit_on_texts(list(x_tr))
#%%
# # Rarewords and its Coverage
x_count, x_totalCount, x_freq, x_totalFreq = rare_words_cover(x_tokenizer, threshold = 4)

#%%
#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer(num_words=x_totalCount - x_count, 
                        filters = token_filter) 
x_tokenizer.fit_on_texts(list(x_tr))

#convert text sequences into integer sequences
x_tr_seq    =   x_tokenizer.texts_to_sequences(x_tr) 
x_val_seq   =   x_tokenizer.texts_to_sequences(x_val)

#padding zero upto maximum length
x_tr    =   pad_sequences(x_tr_seq,  maxlen=MAX_TEXT_LENGTH, padding='post')
x_val   =   pad_sequences(x_val_seq, maxlen=MAX_TEXT_LENGTH, padding='post')

#size of vocabulary ( +1 for padding token)
x_voc   =  x_tokenizer.num_words + 1

#%%
# # Summary Tokenizer
y_raw = y_tr.copy() # Make a copy to check later by string

#prepare a tokenizer for reviews the frequence of words
y_tokenizer = Tokenizer(filters = token_filter)   
y_tokenizer.fit_on_texts(list(y_tr))


# # Rarewords and its Coverage
#%%
y_count, y_totalCount, y_freq, y_totalFreq = rare_words_cover(y_tokenizer, threshold = 3)

# In[37]:


#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer(num_words=y_totalCount-y_count, filters = token_filter) 
y_tokenizer.fit_on_texts(list(y_tr))

#convert text sequences into integer sequences
y_tr_seq    =   y_tokenizer.texts_to_sequences(y_tr) 
y_val_seq   =   y_tokenizer.texts_to_sequences(y_val) 

#padding zero upto maximum length
y_tr    =   pad_sequences(y_tr_seq, maxlen=MAX_SUMMARY_LENGTH, padding='post')
y_val   =   pad_sequences(y_val_seq, maxlen=MAX_SUMMARY_LENGTH, padding='post')

#size of vocabulary
y_voc  =   y_tokenizer.num_words +1

#%%
# Let us check whether word count of start token is equal to length of the training data
y_tokenizer.word_counts['sossonnm'], len(y_tr) 

#%%
# Here, I am deleting the rows that contain only **START** and **END** tokens

y_tr, x_tr = del_if_markOnly(y_tr, x_tr)

y_val, x_val = del_if_markOnly(y_val, x_val)

#%%
# # Model building
# 
# We are finally at the model building part. But before we do that, we need to familiarize ourselves with a few terms which are required prior to building the model.
# 
# **Return Sequences = True**: When the return sequences parameter is set to True, LSTM produces the hidden state and cell state for every timestep
# 
# **Return State = True**: When return state = True, LSTM produces the hidden state and cell state of the last timestep only
# 
# **Initial State**: This is used to initialize the internal states of the LSTM for the first timestep
# 
# **Stacked LSTM**: Stacked LSTM has multiple layers of LSTM stacked on top of each other. 
# This leads to a better representation of the sequence. I encourage you to experiment with the multiple layers of the LSTM stacked on top of each other (it’s a great way to learn this)
# 
# Here, we are building a 3 stacked LSTM for the encoder:


K.clear_session()

latent_dim = 340
embedding_dim = 100

# Encoder
encoder_inputs = Input(shape=(MAX_TEXT_LENGTH,))

#embedding layer
enc_emb =  Embedding(x_voc, embedding_dim, trainable=True)(encoder_inputs)

#encoder lstm 1
encoder_lstm1 = Bidirectional(LSTM(latent_dim,return_sequences=True,
                                    return_state=True,dropout=0.4,recurrent_dropout=0))

encoder_output1, state_h1_f, state_c1_f, state_h1_b, state_c1_b = encoder_lstm1(enc_emb)

#encoder lstm 2
encoder_lstm2 = Bidirectional(LSTM(latent_dim,return_sequences=True,
                                   return_state=True,dropout=0.4,recurrent_dropout=0))
encoder_output2, state_h2_f, state_c2_f, state_h2_b, state_c2_b = encoder_lstm2(encoder_output1)

#encoder lstm 3
encoder_lstm3 = Bidirectional(LSTM(latent_dim,return_sequences=True,
                                   return_state=True,dropout=0.4,recurrent_dropout=0))
encoder_outputs, state_h_f, state_c_f, state_h_b, state_c_b = encoder_lstm3(encoder_output2)

state_h = Concatenate()([state_h_f, state_h_b])
state_c = Concatenate()([state_c_f, state_c_b])
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

#embedding layer
dec_emb_layer = Embedding(y_voc, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim,return_sequences=True,
                                   return_state=True,dropout=0.4,recurrent_dropout=0)

decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state = encoder_states)

# Attention layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# Concat attention input and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#dense layer
decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary() 

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)

history=model.fit([x_tr,y_tr[:,:-1]], 
                  y_tr.reshape(y_tr.shape[0],
                               y_tr.shape[1], 1)[:,1:],
                  epochs=20,
                  callbacks=[es],
                  batch_size=4, 
                  validation_data=([x_val,y_val[:,:-1]], 
                                   y_val.reshape(y_val.shape[0],
                                                 y_val.shape[1], 1)[:,1:]))


# # Understanding the Diagnostic plot
# 
# Now, we will plot a few diagnostic plots to understand the behavior of the model over time:

# In[ ]:


from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('loss_curve.png')


# From the plot, we can infer that validation loss has increased after epoch 17 for 2 successive epochs. Hence, training is stopped at epoch 19.
# 
# Next, let’s build the dictionary to convert the index to word for target and source vocabulary:

# In[ ]:


reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index


# # Inference
# 
# Set up the inference for the encoder and decoder:

# In[ ]:


# Encode the input sequence to get the feature vector
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_text_len, latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs) 
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

#attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat) 

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])


# We are defining a function below which is the implementation of the inference process (which we covered [here](https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/)):

# In[ ]:


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


# Let us define the functions to convert an integer sequence to a word sequence 
# for summary as well as the reviews:

# In[ ]:


def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString


# Here are a few summaries generated by the model:

# In[ ]:

full_text = []
origin_sum = []
machine_sum = []
for i in range(0,100):
    print("Review:",seq2text(x_tr[i]))
    full_text.append(seq2text(x_tr[i]))
    print("Original summary:",y_raw[i])
    origin_sum.append(seq2summary(y_tr[i]))
    print("Predicted summary:",decode_sequence(x_tr[i].reshape(1,max_text_len)))
    machine_sum.append(decode_sequence(x_tr[i].reshape(1,max_text_len)))
    print("\n")

result = pd.DataFrame({'fullText': full_text, 
                       'originSum': origin_sum,
                       'machineSum': machine_sum})
result['originSum'] = result['originSum'].apply(lambda x :  x.replace(' eostok','').replace('sostok ', ''))
result.to_csv('./data/predict_0.csv')
#%%


# This is really cool stuff. Even though the actual summary and the summary generated by our model do not match in terms of words, both of them are conveying the same meaning. Our model is able to generate a legible summary based on the context present in the text.
# 
# This is how we can perform text summarization using deep learning concepts in Python.
# 
# #How can we Improve the Model’s Performance Even Further?
# 
# Your learning doesn’t stop here! There’s a lot more you can do to play around and experiment with the model:
# 
# I recommend you to **increase the training dataset** size and build the model. The generalization capability of a deep learning model enhances with an increase in the training dataset size
# 
# Try implementing **Bi-Directional LSTM** which is capable of capturing the context from both the directions and results in a better context vector
# 
# Use the **beam search strategy** for decoding the test sequence instead of using the greedy approach (argmax)
# 
# Evaluate the performance of your model based on the **BLEU score**
# 
# Implement **pointer-generator networks** and **coverage mechanisms**
#  
# 
# 

# #End Notes
# 
# If you have any feedback on this article or any doubts/queries, kindly share them in the comments section over [here](https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/) and I will get back to you. And make sure you experiment with the model we built here and share your results with me!
