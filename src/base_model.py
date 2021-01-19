# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 21:13:07 2021

@author: NguyenSon
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Concatenate, \
                                    Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from src.attention import AttentionLayer

def base_model_0(x_vocab_size, y_vocab_size, latent_dim, embedding_dim, max_text_length, is_save = False):
    """
    Model with 3 layers LSTM in encoder 

    Parameters
    ----------
    x_vocab_size : TYPE int
        DESCRIPTION. Size of the input vocabulary
    y_vocab_size : TYPE int
        DESCRIPTION. Size of the summarized vocabulary
    latent_dim : TYPE int
        DESCRIPTION. Number of nodes in 1 layer 
    embedding_dim : TYPE int
        DESCRIPTION. dimension for embedding vector
    max_text_length : TYPE int
        DESCRIPTION. number of words in the input text
    is_save : TYPE bool, optional 
        DESCRIPTION. for choosing whether to save the model or not. The default is False.

    Returns
    -------
    full_model : TYPE object
        DESCRIPTION. full model
    encoder_model : TYPE object
        DESCRIPTION. encoder model for the inference
    decoder_model : TYPE object
        DESCRIPTION. decoder model for the inference

    """
    # Encoder
    encoder_inputs = Input(shape=(max_text_length,))
    
    enc_emb =  Embedding(x_vocab_size, embedding_dim, trainable=True)(encoder_inputs)
    
    #encoder lstm 1
    encoder_lstm1 = LSTM(latent_dim,
                         return_sequences=True,
                         return_state=True,
                         dropout=0.4)
    
    encoder_outputs1, state_h1, state_c1 = encoder_lstm1(enc_emb)
    #encoder lstm 2
    encoder_lstm2 = LSTM(latent_dim,
                         return_sequences=True,
                         return_state=True,
                         dropout=0.4)
    encoder_outputs2, state_h2, state_c2 = encoder_lstm2(encoder_outputs1)
    
    #encoder lstm 3
    encoder_lstm3 = LSTM(latent_dim,
                         return_sequences=True,
                         return_state=True,
                         dropout=0.4)
    
    encoder_outputs, state_h, state_c = encoder_lstm3(encoder_outputs2)
    
    # Decoder
    decoder_inputs = Input(shape=(None,))
    
    dec_emb_layer = Embedding(y_vocab_size, embedding_dim, trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)
    
    decoder_lstm = LSTM(latent_dim,
                        return_sequences=True,
                        return_state=True,
                        dropout=0.4)
    
    decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state = [state_h, state_c])
    
    # Attention layer
    attn_layer = AttentionLayer(name='Attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
    
    # Concat attention input and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
    
    # Last dense layer
    decoder_dense =  TimeDistributed(Dense(y_vocab_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)
    
    # Define the model 
    full_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    full_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    print('________________________FULL MODEL SUMMARY_______________________')
    print(full_model.summary())
    
    # Inference Model
    # Encoder Model
    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])
    print('________________________INFERENCE ENCODER MODEL SUMMARY_______________________')
    print(encoder_model.summary())
    
    # Decoder Model
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(max_text_length, latent_dim))
    
    dec_emb2 = dec_emb_layer(decoder_inputs) 
    
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, 
                                                        initial_state=[decoder_state_input_h, decoder_state_input_c])
    

    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])
    
    # Output layer for probability of each word in vocab
    decoder_outputs2 = decoder_dense(decoder_inf_concat) 
    
    # Define Inference Decoder
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])
    print('________________________INFERENCE DECODER MODEL SUMMARY_______________________')
    print(decoder_model.summary())
    
    if is_save:
        full_model.save('./models/3_LSTM_model_train')
        encoder_model.save('./models/3_LSTM_encoder_model')
        decoder_model.save('./models/3_LSTM_decoder_model')
    
    return full_model, encoder_model, decoder_model


def base_model_1(x_vocab_size, y_vocab_size, latent_dim, embedding_dim, max_text_length, is_save = False):
    """
    Model with 1 layers BiLSTM + 1 layers LSTM in encoder 

    Parameters
    ----------
    x_vocab_size : TYPE int
        DESCRIPTION. Size of the input vocabulary
    y_vocab_size : TYPE int
        DESCRIPTION. Size of the summarized vocabulary
    latent_dim : TYPE int
        DESCRIPTION. Number of nodes in 1 layer 
    embedding_dim : TYPE int
        DESCRIPTION. dimension for embedding vector
    max_text_length : TYPE int
        DESCRIPTION. number of words in the input text
    is_save : TYPE bool, optional 
        DESCRIPTION. for choosing whether to save the model or not. The default is False.

    Returns
    -------
    full_model : TYPE object
        DESCRIPTION. full model
    encoder_model : TYPE object
        DESCRIPTION. encoder model for the inference
    decoder_model : TYPE object
        DESCRIPTION. decoder model for the inference

    """
    # Encoder
    encoder_inputs = Input(shape=(max_text_length,))
    
    enc_emb =  Embedding(x_vocab_size, embedding_dim, trainable=True)(encoder_inputs)
    
    #encoder bilstm 1
    encoder_bilstm1 = Bidirectional(LSTM(latent_dim,
                                         return_sequences=True,
                                         return_state=True,
                                         dropout=0.4))

    encoder_outputs1, state_h1_f, state_c1_f, state_h1_b, state_c1_b = encoder_bilstm1(enc_emb)
    
    #encoder lstm 2
    encoder_lstm2 = LSTM(latent_dim,
                         return_sequences=True,
                         return_state=True,
                         dropout=0.4)
    encoder_outputs, state_h, state_c = encoder_lstm2(encoder_outputs1)

    
    # Decoder
    decoder_inputs = Input(shape=(None,))
    
    dec_emb_layer = Embedding(y_vocab_size, embedding_dim, trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)
    
    decoder_lstm = LSTM(latent_dim,
                        return_sequences=True,
                        return_state=True,
                        dropout=0.4)
    
    decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state = [state_h, state_c])
    
    # Attention layer
    attn_layer = AttentionLayer(name='Attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
    
    # Concat attention input and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
    
    # Last dense layer
    decoder_dense =  TimeDistributed(Dense(y_vocab_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)
    
    # Define the model 
    full_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    full_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    print('________________________FULL MODEL SUMMARY_______________________')
    print(full_model.summary())
    
    # Inference Model
    # Encoder Model
    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])
    print('________________________INFERENCE ENCODER MODEL SUMMARY_______________________')
    print(encoder_model.summary())
    
    # Decoder Model
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(max_text_length, latent_dim))
    
    dec_emb2 = dec_emb_layer(decoder_inputs) 
    
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, 
                                                        initial_state=[decoder_state_input_h, decoder_state_input_c])
    

    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])
    
    # Output layer for probability of each word in vocab
    decoder_outputs2 = decoder_dense(decoder_inf_concat) 
    
    # Define Inference Decoder
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])
    print('________________________INFERENCE DECODER MODEL SUMMARY_______________________')
    print(decoder_model.summary())
    
    if is_save:
        full_model.save('./models/11_Bi_LSTM_model_train')
        encoder_model.save('./models/11_Bi_LSTM_encoder_model')
        decoder_model.save('./models/11_Bi_LSTM_decoder_model')
    
    return full_model, encoder_model, decoder_model

def base_model_2(x_vocab_size, y_vocab_size, latent_dim, embedding_dim, max_text_length, is_save = False):
    """
    Model with 3 layers BiLSTM in encoder 

    Parameters
    ----------
    x_vocab_size : TYPE int
        DESCRIPTION. Size of the input vocabulary
    y_vocab_size : TYPE int
        DESCRIPTION. Size of the summarized vocabulary
    latent_dim : TYPE int
        DESCRIPTION. Number of nodes in 1 layer 
    embedding_dim : TYPE int
        DESCRIPTION. dimension for embedding vector
    max_text_length : TYPE int
        DESCRIPTION. number of words in the input text
    is_save : TYPE bool, optional 
        DESCRIPTION. for choosing whether to save the model or not. The default is False.

    Returns
    -------
    full_model : TYPE object
        DESCRIPTION. full model
    encoder_model : TYPE object
        DESCRIPTION. encoder model for the inference
    decoder_model : TYPE object
        DESCRIPTION. decoder model for the inference

    """
    # Encoder
    encoder_inputs = Input(shape=(max_text_length,))
    
    enc_emb =  Embedding(x_vocab_size, embedding_dim, trainable=True)(encoder_inputs)
    
    # encoder bilstm 1
    encoder_bilstm1 = Bidirectional(LSTM(latent_dim,
                                       return_sequences=True,
                                       return_state=True,
                                       dropout=0.4))

    encoder_outputs1, state_h1_f, state_c1_f, state_h1_b, state_c1_b = encoder_bilstm1(enc_emb)
    
    # encoder bilstm 2
    encoder_bilstm2 = Bidirectional(LSTM(latent_dim,
                                       return_sequences=True,
                                       return_state=True,
                                       dropout=0.4))

    encoder_outputs2, state_h2_f, state_c2_f, state_h2_b, state_c2_b = encoder_bilstm2(encoder_outputs1)
    
    # encoder bilstm 3
    encoder_bilstm3 = Bidirectional(LSTM(latent_dim,
                                       return_sequences=True,
                                       return_state=True,
                                       dropout=0.4))

    encoder_outputs, state_h_f, state_c_f, state_h_b, state_c_b = encoder_bilstm3(encoder_outputs2)
    
    state_h = Concatenate()([state_h_f, state_h_b])
    state_c = Concatenate()([state_c_f, state_c_b])
    
    # Decoder
    decoder_inputs = Input(shape=(None,))
    
    dec_emb_layer = Embedding(y_vocab_size, embedding_dim, trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)
    
    decoder_lstm = LSTM(latent_dim*2,
                        return_sequences=True,
                        return_state=True,
                        dropout=0.4)
    
    decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state = [state_h, state_c])
    
    # Attention layer
    attn_layer = AttentionLayer(name='Attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
    
    # Concat attention input and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
    
    # Last dense layer
    decoder_dense =  TimeDistributed(Dense(y_vocab_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)
    
    # Define the model 
    full_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    full_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    print('________________________FULL MODEL SUMMARY_______________________')
    print(full_model.summary())
    
    """Inference Model"""
    # Encoder Model
    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, 
                                                         state_h_f, state_c_f,
                                                         state_h_b, state_c_b])
    print('________________________INFERENCE ENCODER MODEL SUMMARY_______________________')
    print(encoder_model.summary())
    
    # Decoder Model
    decoder_state_input_h = Input(shape=(latent_dim*2,))
    decoder_state_input_c = Input(shape=(latent_dim*2,))
    decoder_hidden_state_input = Input(shape=(max_text_length, latent_dim*2))
    
    dec_emb2 = dec_emb_layer(decoder_inputs) 
    
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, 
                                                        initial_state=[decoder_state_input_h, decoder_state_input_c])
    

    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])
    
    # Output layer for probability of each word in vocab
    decoder_outputs2 = decoder_dense(decoder_inf_concat) 
    
    # Define Inference Decoder
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])
    print('________________________INFERENCE DECODER MODEL SUMMARY_______________________')
    print(decoder_model.summary())
    
    if is_save:
        full_model.save('./models/3_BiLSTM_model_train')
        encoder_model.save('./models/3_BiLSTM_encoder_model')
        decoder_model.save('./models/3_BiLSTM_decoder_model')
    
    return full_model, encoder_model, decoder_model

# full_model, encoder_model, decoder_model = base_model_2(4000,3000,340,100,200)