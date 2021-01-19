
import numpy as np
from tensorflow.keras.layers import Concatenate

def decode_sequence(encoder_model, decoder_model, 
                    target_word_index, reverse_target_word_index,
                    input_seq, max_summary_length, is_Bi = False):
    """
    Generate Summarized text

    Parameters
    ----------
    encoder_model : TYPE object
        DESCRIPTION. Encoder model for inference
    decoder_model : TYPE object
        DESCRIPTION. Decode model for inference
    target_word_index : TYPE object
        DESCRIPTION. tokenizer.word_index (tokenizer from keras)
    reverse_target_word_index : TYPE object
        DESCRIPTION. tokenizer.index_word for target dataset
    input_seq : TYPE string
        DESCRIPTION. input of the new text or article
    max_summary_length : TYPE int
        DESCRIPTION. defined max length of the summary
    is_Bi : TYPE bool, optional
        DESCRIPTION. is the encoder is bidirectional or not. The default is False.

    Returns
    -------
    decoded_sentence : TYPE string
        DESCRIPTION. generated summary sequence

    """

    if is_Bi:
        # Encode for BiLSTM output
        e_out, e_h_f, e_c_f, e_h_b, e_c_b = encoder_model.predict(input_seq)
        
        e_h = Concatenate()([e_h_f, e_h_b])
        e_c = Concatenate()([e_c_f, e_c_b])
    else:
        # Encode for LSTM output
        e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sossonnm']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        try:
            sampled_token = reverse_target_word_index[sampled_token_index]
        except:
            print('__________________________________________')
            print('Token index {} is not found in the tokenizer'.format(sampled_token_index))
            print('Using the second highest probability value instead')
            sampled_token_index = np.argsort(output_tokens[0, -1, :])[-2]
            sampled_token = reverse_target_word_index[sampled_token_index]
            print('Sampled token: {}'.format(sampled_token))
            print('__________________________________________')
        
        if(sampled_token!='eossonnm'):
            decoded_sentence += ' '+ sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eossonnm'  or len(decoded_sentence.split()) >= (max_summary_length-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq, target_word_index, reverse_target_word_index):
    """
    tokenized sequence to string

    Parameters
    ----------
    input_seq : TYPE object
        DESCRIPTION. summary of the dataset after padding
    target_word_index : TYPE object
        DESCRIPTION. tokenizer.word_index (tokenizer from keras)
    reverse_target_word_index : TYPE object
        DESCRIPTION. tokenizer.index_word of target dataset (tokenizer from keras)

    Returns
    -------
    newString : TYPE string
        DESCRIPTION. reverted from padding to string original summary

    """
    newString = ''
    for i in input_seq:
        if((i != 0 and i != target_word_index['sossonnm']) and i != target_word_index['eossonnm']):
            newString = newString + reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq, reverse_source_word_index):
    """
    tokenized sequence to string

    Parameters
    ----------
    input_seq : TYPE object
        DESCRIPTION. predicted summary of the dataset after decoding from the inference model
    reverse_source_word_index : TYPE object
        DESCRIPTION. tokenizer.index_word of fullText dataset (tokenizer from keras)

    Returns
    -------
    newString : TYPE string
        DESCRIPTION. reverted from padding to string original full text

    """
    newString=''
    for i in input_seq:
        if(i != 0):
            newString = newString + reverse_source_word_index[i]+' '
    return newString