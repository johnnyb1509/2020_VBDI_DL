# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:06:39 2021

@author: sonnm12
"""

import numpy as np


# greedy decoder
def greedy_decoder(data):
	# index for largest probability each row
	return [np.argmax(s) for s in data]
 
# define a sequence of 10 words over a vocab of 5 words
data = [[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1]]
data = np.array(data)
# decode sequence
result = greedy_decoder(data)
print(result)
#%%
def beam_search_decoder(data, k):
    sequences = [[list(), 0.0]]
	# walk over each step in sequence
    for row in data:
        print(row)
        all_candidates = list()
		# expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            print(score)
            for j in range(len(row)):
                candidate = [seq + [j], score - np.log(row[j])]
                all_candidates.append(candidate)
		# order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
        sequences = ordered[:k]
        print(sequences)
    return sequences, all_candidates

beam_result, all_candidate = beam_search_decoder(data, 4)