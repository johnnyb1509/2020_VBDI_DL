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
        - beam_search.py            # Beam search self-created (still in testing stage)
        - inference_summary.py      # Function to rewrite the summary from full text
        - rouge_score.py            # Rouge score
        - utils_.py                 # other functions needed for project
    - trainning.py                  # for training
    - decoder.py                    # for decoding and calculate rouge score
    - preprocess_data.ipynb         # Data preprocessor for each kind of model
    - evaluate.ipynb                # visualize the rouge score of result
    - requirement.txt
```

