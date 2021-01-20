# 2020_VBDI_DL
VinBigData_Deeplearning

### Member:
- Nguyen Minh Son
- Dang Quynh Anh
- Le Ngoc Hai

### Construction of code:
```
- project\
    - src\
        - attention.py              # Attention module
        - base_model.py             # Include 3 base models
        - beam_search.py            # Self-created Beam search  (still in testing stage)
        - inference_summary.py      # Function to rewrite the summary from full text
        - rouge_score.py            # Rouge score
        - utils_.py                 # other functions needed for project
    - trainning.py                  # for training
    - decoder.py                    # for decoding and calculate rouge score
    - preprocess_data.ipynb         # Data preprocessor for each kind of model
    - evaluate.ipynb                # visualize the rouge score of result
    - requirement.txt
```
### Data
- Data used in this project is re-format and uploaded from the MediaFire link:
https://www.mediafire.com/file/ikhwaqjaixsb0qf/data_base_model.rar/file

In which, the raw data is the Vietnamese Benchmark Data VNDS:
- DOI: 10.1109/NICS48868.2019.9023886
- Github: https://github.com/ThanhChinhBK/vietnews

### Referrence SOTA models used in this project
#### FastAbs 
We borrowed source code from Yen-Chun Chen, Mohit Bansal to re-train:
- Link: https://github.com/ChenRocks/fast_abs_rl
The code is used in his paper "Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting"(https://arxiv.org/abs/1805.11080)
#### BERTSUM 
We borrowed source code from Yang Liu to re-build our model for Vietnamese Extractive summary:
- Link: https://github.com/nlpyang/BertSum
The code is used in his paper "Fine-tune BERT for Extractive Summarization"(https://arxiv.org/pdf/1903.10318.pdf)

