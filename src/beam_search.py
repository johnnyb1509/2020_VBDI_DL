# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 02:06:56 2021

@author: NguyenSon
"""

import numpy as np
import math

# beam_w  = 5
# stop_condition = False
# decoded_sentence = ''
# beam_decoder = Beam_Decoder(x_tokenizer, y_tokenizer, 
#                             decoder_model, e_out, 
#                             MAX_SUMMARY_LENGTH, beam_width = beam_w)
# output_tokens, h, c = beam_decoder.predict(target_seq, e_h, e_c)
# idx_array, prob_array = beam_decoder.beam_filter(output_tokens)
# # sentence_idx, last_target_seq, last_sum_prob, hidden_save, cell_save, out_save  = beam_decoder.next_(idx_array, prob_array, h, c)
# # output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

# stop_condition = False
# i = 0
# # while not stop_condition:
# for i_ in range(8):
#     i += 1
#     print('=================TURN', i)
#     sentence_idx, last_target_seq, last_sum_prob, hidden_save, cell_save, out_save, score_all  = beam_decoder.next_(idx_array, prob_array, h, c)

#     h, c = hidden_save, cell_save
#     idx_array = last_target_seq
#     beam_decoder.update_best_sentence(sentence_idx)
    
#     for num_sen in range(beam_w):
        
#         if beam_decoder.sentence_length['sen{}'.format(num_sen)] >= MAX_SUMMARY_LENGTH:
#             stop_condition = True
    

class Beam_Decoder():
    def __init__(self, x_tokenizer, y_tokenizer,
                         decoder_model, encoder_output, max_summary_len,
                         beam_width = 5):
        self.reverse_target_word = y_tokenizer.index_word
        self.reverse_source_word = x_tokenizer.index_word
        self.target_word_index = y_tokenizer.word_index
        
        self.beam_width = beam_width
        self.decoder_model = decoder_model
    
        self.encoder_output = encoder_output

        
        self.max_summary_len = max_summary_len
        self.score = {}
        self.sentence_save = {'sen0': [],
                            'sen1': [],
                            'sen2': [],
                            'sen3': [],
                            'sen4': [],}
        
        self.sentence_length = {'sen0': [],
                            'sen1': [],
                            'sen2': [],
                            'sen3': [],
                            'sen4': [],}
        return
    
    
    def predict(self, updated_target_seq, updated_hiddenState, updated_cellState):
        output_tokens_cache, last_hiddenState_cache, last_cellState_cache = self.decoder_model.predict([updated_target_seq] + [self.encoder_output, 
                                                                                                                         updated_hiddenState, 
                                                                                                                         updated_cellState])
        return output_tokens_cache, last_hiddenState_cache, last_cellState_cache
     
    def beam_filter(self, ouput_token_step):
        """
        Funtion to filter highest <beam_width> probbability of output and its index in array

        Returns
        -------
        None.

        """
        idx_array = np.argpartition(ouput_token_step[0, -1, :], -self.beam_width)[-self.beam_width:] #index for reverse_target_word
        prob_array = np.log(np.partition(ouput_token_step[0, -1, :], -self.beam_width)[-self.beam_width:])
        
        return idx_array, prob_array
    
    def next_(self, idx_array, prob_array, updated_hiddenState, updated_cellState):
        hidden_save = {}
        cell_save = {}
        out_save = {}
        score_all = {}
        score_list = []
        for i in range(len(idx_array)): # run 1 step forward
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = idx_array[i]
            print("target_seq", idx_array[i])
            if type(updated_hiddenState) != dict:
                out_cache, hidden_cache, cell_cache = self.predict(target_seq, updated_hiddenState, updated_cellState)
            else:
                out_cache, hidden_cache, cell_cache = self.predict(target_seq, list(updated_hiddenState.values())[i], 
                                                                   list(updated_cellState.values())[i])
            idx_array_step, prob_array_step = self.beam_filter(out_cache) # collect highest probability to reduce calculation volume
            print('idx_after_beam', idx_array_step)
            for i_step in range(len(idx_array_step)): # calculate sum of log probability 
                hidden_save[idx_array[i],idx_array_step[i_step]] = hidden_cache
                cell_save[idx_array[i],idx_array_step[i_step]] = cell_cache
                out_save[idx_array[i],idx_array_step[i_step]] = out_cache[0, -1, :]
                
                scr = prob_array[i] + prob_array_step[i_step]
                score_all[idx_array[i],idx_array_step[i_step]] = scr # score for each pair (prior, post)
                score_list.append(scr) # get all score
                
                print(idx_array[i], idx_array_step[i_step])
                
            print('______________________')
        max_width_prob = np.partition(score_list, -self.beam_width)[-self.beam_width:] # collect the highest sum of log probability 
        print(max_width_prob)
        print(score_all)
        sentence_idx, last_target_seq, last_sum_prob = self.get_vocab_key(score_all, max_width_prob) # filter the pair with highest sum
        print('>>>>>>>>>Choosing>>>>>>>>>>>', sentence_idx)
        hidden_save_r = {k:hidden_save[k] for k in sentence_idx} # saving hidden state
        cell_save_r = {k:cell_save[k] for k in sentence_idx} # saving cell state
        out_save_r = {k:out_save[k] for k in sentence_idx} # saving result 
        return sentence_idx, last_target_seq, last_sum_prob, hidden_save_r, cell_save_r, out_save_r, score_all
    
    def get_vocab_key(self, dict_score, ref_list):
        sentence_idx_list = []
        last_target_seq = []
        last_sum_prob = []
        for val in ref_list:
            for key, value in dict_score.items():
                if value == val:
                    sentence_idx_list.append(key)
                    last_target_seq.append(key[-1])
                    last_sum_prob.append(value)
        print(last_sum_prob)
        return sentence_idx_list, last_target_seq, last_sum_prob
                    
    def check_end_mark(self, last_idx):
        if self.reverse_target_word[last_idx] == 'eossonnm':
            return True
        else:
            return False
    
    def NestedDictValues(d):
        for v in d.values():
            if isinstance(v, dict):
                yield from NestedDictValues(v)
            else:
                yield v
        
    def update_best_sentence(self, new_sentence_idx):
        for i in range(len(new_sentence_idx)):
            prior = new_sentence_idx[i][0]
            for j in range(5):
                post = self.sentence_save['sen{}'.format(j)](-1)
                if post == prior:
                    self.sentence_save['sen{}'.format(i)].append(new_sentence_idx[i])
            self.sentence_length['sen{}'.format(i)] = len(self.sentence_save['sen{}'.format(i)])
        pass





def beam_search_decoder(predictions, top_k = 3):
    #start with an empty sequence with zero score
    output_sequences = [([], 0)]
    
    #looping through all the predictions
    for token_probs in predictions:
        new_sequences = []
        
        #append new tokens to old sequences and re-score
        for old_seq, old_score in output_sequences:
            for char_index in range(len(token_probs)):
                new_seq = old_seq + [char_index]
                #considering log-likelihood for scoring
                new_score = old_score + math.log(token_probs[char_index])
                new_sequences.append((new_seq, new_score))
                
        #sort all new sequences in the de-creasing order of their score
        output_sequences = sorted(new_sequences, key = lambda val: val[1], reverse = True)
        
        #select top-k based on score 
        # *Note- best sequence is with the highest score
        output_sequences = output_sequences[:top_k]
        
    return output_sequences