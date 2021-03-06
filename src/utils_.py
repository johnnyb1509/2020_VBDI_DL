
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

def data_reader(data_dir, column_text, column_summary, max_text_length, max_summary_length):
    """
    Reading - Cleaning - Filtering - Marking data

    Parameters
    ----------
    data_dir : TYPE string
        DESCRIPTION. Address of data directory
    column_text, column_summary : TYPE string
        DESCRIPTION. columns name using in the process 
    max_text_length : TYPE int 
        DESCRIPTION. desired maximum words in the text
    max_summary_length : TYPE int
        DESCRIPTION. desired maximum words in the summary

    Returns
    -------
    x_data : TYPE array
        DESCRIPTION. array of Full text data
    y_data : TYPE array
        DESCRIPTION. array of summary data

    """
   
    data_ = pd.read_csv(data_dir, usecols=[column_text, column_summary]) #Read data
    data_ = del_short_summary(data_, column_summary) # delete short summary (lower than 3 words)
    data_ = frame_clean(data_, column_text, column_summary) # cleaning frame
    data_ = filter_data_length(data_, 'cleaned_text', 'cleaned_summary', 
                               max_text_length, max_summary_length) #Filter data on desired length
    data_ = mark_sentence(data_, 'cleaned_summary') # marking <sos> and <eos>
    
    x_data = np.array(data_['cleaned_text'])
    y_data = np.array(data_['cleaned_summary'])
    return x_data, y_data

def loss_curve(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig('loss_curve.png')
    return

def del_short_summary(df_raw, summary_col, threshold = 3):
    """
    Delete short summary under number of words

    Parameters
    ----------
    df_raw : TYPE DataFrame
        DESCRIPTION. Frame of segmented and cleaned data
    summary_col : TYPE string
        DESCRIPTION. Name of column contains summary text
    threshold : TYPE int , optional
        DESCRIPTION. The default is 3. Min number of word in summary 

    Returns
    -------
    df : TYPE DataFrame
        DESCRIPTION. New frame that drop all too short summary

    """
    
    df = df_raw.copy()
    leng_sum =[]
    for i,r in df.iterrows():
        leng_sum.append(len(r[summary_col].split()))
    df['len_sum'] = leng_sum
    df = df[df['len_sum'] > threshold]
    return df

def words_dist(df, fullText_col, summary_col, is_save = True):
    """
    Plot the distribution of words in a certain dataset

    Parameters
    ----------
    df : TYPE DataFrame
        DESCRIPTION. Frame contains cleaned corpus
    fullText_col : TYPE string
        DESCRIPTION. name of the full text column
    summary_col : TYPE string
        DESCRIPTION. name of the summary column

    Returns
    -------
    None.
    Graph ploted

    """
    text_word_count = []
    summary_word_count = []
    for i in df[fullText_col]:
          text_word_count.append(len(i.split()))
    
    for i in df[summary_col]:
          summary_word_count.append(len(i.split()))
    
    length_df = pd.DataFrame({'Raw_Text':text_word_count, 'Summary':summary_word_count})
    length_df.hist(bins = 30)
    # plt.title('Words Distribution')
    plt.show()
    if is_save:
        plt.savefig('./words_distribution.png')
        
    return

def text_cleaner(text):
    """
    Text Cleaner for cleaning some syntax

    Parameters
    ----------
    text : TYPE string
        DESCRIPTION. corpus 

    Returns
    -------
    newString : TYPE string
        DESCRIPTION. cleaned corpus

    """
    newString = text.replace('(', ' ').replace(')',' ').replace('.', ' ').replace(',', ' ')
    newString = newString.replace('/', ' ').replace('%', ' ').replace(';', ' ').replace('’', ' ')
    newString = newString.replace('“', ' ').replace('”', ' ').replace('"', ' ').replace("'", ' ')
    try:
        newString = newString.lower()
    except:
        print("Check for lower the string, maybe it just a number")
    return newString

def frame_clean(df, text_col, summary_col):
    cleaned_text = []
    cleaned_summary = []
    for i, r in df.iterrows():
        print('Cleaning row ', i)
        cleaned_text.append(text_cleaner(r[text_col]))
        cleaned_summary.append(text_cleaner(r[summary_col]))

    data_cleaned = pd.DataFrame()
    data_cleaned['cleaned_text'] = cleaned_text
    data_cleaned['cleaned_summary'] = cleaned_summary
    return data_cleaned

def filter_data_length(df_raw, cleaned_text_col, cleaned_summary_col,
                       text_len, summary_len):
    """
    Shorten the length of given corpus

    Parameters
    ----------
    df : TYPE DataFrame
        DESCRIPTION. frame of the dataset
    cleaned_text_col : TYPE string
        DESCRIPTION. name of text column
    cleaned_summary_col : TYPE string
        DESCRIPTION. name of summary col
    text_len : TYPE int
        DESCRIPTION. MAX_TEXT_LENGTH
    summary_len : TYPE int
        DESCRIPTION. MAX_SUMMARY_LENGTH

    Returns
    -------
    df : TYPE DataFrame
        DESCRIPTION. New dataframe with desired length of 
                    fullText and summary
    """
    cleaned_text =np.array(df_raw[cleaned_text_col])
    cleaned_summary=np.array(df_raw[cleaned_summary_col])
    
    short_text=[]
    short_summary=[]
    
    for i in range(len(cleaned_text)):
        if(len(cleaned_summary[i].split()) <= summary_len and \
           len(cleaned_text[i].split()) <= text_len):
            short_text.append(cleaned_text[i])
            short_summary.append(cleaned_summary[i])

    df=pd.DataFrame({cleaned_text_col:short_text,
                     cleaned_summary_col:short_summary})
    return df

def mark_sentence(df, summary_col):
    """
    Mark summary sentences for the Decoder
    <sos> Start Of Sentence
    <eos> End Of Sentence

    Parameters
    ----------
    df : TYPE dataframe
        DESCRIPTION. 
    summary_col : TYPE string
        DESCRIPTION. name of the summary column

    Returns
    -------
    df : TYPE dataframe
        DESCRIPTION. Frame with summary data marked <sos> .... <eos>

    """
    df[summary_col] = df[summary_col].apply(lambda x : 'sossonnm '+ x + ' eossonnm')
    return df

def percent_count(df, column_name):
    """
    Count percentage for choosing reasonable maximum length of text(or summary)

    Parameters
    ----------
    df : TYPE DataFrame
        DESCRIPTION. 
    column_name : TYPE string
        DESCRIPTION. column to check

    Returns
    -------
    None.

    """
    cnt=0
    for i in df[column_name]:
        if(len(i.split())<=30):
            cnt=cnt+1
    print('% of max len text on all text: {}%'.format(round(cnt/len(df[column_name])*100), 3))
    return

def rare_words_cover(factor_tokenized, threshold = 4):
    """
    Calculate the counting and frequency of the rare words in given tokenized 

    Parameters
    ----------
    factor_tokenized : TYPE object
        DESCRIPTION. object after Tokenizer().fit_on_texts()
    thresh : TYPE int, optional
        DESCRIPTION. Number of words that defined as rare words. The default is 4. 
        
    Returns
    -------
    cnt : TYPE int
        DESCRIPTION. Count on number of words that have total of appearance under given threshold
    tot_cnt : TYPE int
        DESCRIPTION. total number of UNIQUE words in tokenizer
    freq : TYPE float
        DESCRIPTION. Appearing frequency of words 
    tot_freq : TYPE float
        DESCRIPTION. Total of appearance. This is the sum up 

    """
    count=0
    total_count=0
    freq=0
    total_freq=0
    
    for key,value in factor_tokenized.word_counts.items():
        total_count=total_count + 1 # count all
        total_freq=total_freq + value
        if(value < threshold):
            count = count + 1 # count rare
            freq = freq + value
        
    print("% of rare words in vocabulary:",(count/total_count)*100)
    print("Total Coverage of rare words:",(freq/total_freq)*100)
    return count, total_count, freq, total_freq

def del_if_markOnly(seq_matrix_target, seq_matrix_features):
    """
    Function for deleting the summary that has only mark 'sos' and 'eos'

    Parameters
    ----------
    seq_matrix_target : TYPE matrix
        DESCRIPTION. matrix of target that has been tokenized and padded
    seq_matrix_features : TYPE matrix
        DESCRIPTION. matrix of features that has been tokenized and padded

    Returns
    -------
    seq_matrix_target : TYPE
        DESCRIPTION.
    seq_matrix_features : TYPE
        DESCRIPTION.

    """
    idx=[]
    for i in range(len(seq_matrix_target)):
        count=0
        for j in seq_matrix_target[i]:
            if j!=0:
                count=count+1
        if(count==2):
            idx.append(i)
    
    seq_matrix_target = np.delete(seq_matrix_target, idx, axis=0)
    seq_matrix_features = np.delete(seq_matrix_features, idx, axis=0)
    return seq_matrix_target, seq_matrix_features

def tokenizer(train, validate, max_text_length, threshold = 4):
    """
    Tokenizing the given dataset

    Parameters
    ----------
    train : TYPE array
        DESCRIPTION. train dataset
    validate : TYPE array
        DESCRIPTION. validate dataset
    max_text_length : TYPE int
        DESCRIPTION. defined maximum text length (counting number of words in corpus)
    threshold : TYPE int, optional
        DESCRIPTION. the numbers of appearance of words 
                    Defining for how much frequency is count to rare. The default is 4.

    Returns
    -------
    train : TYPE object
        DESCRIPTION. tokenized data train
    validate : TYPE object
        DESCRIPTION. tokenized data validate
    vocab_size : TYPE int
        DESCRIPTION. size of the vocabulary of the dataset
    tokenizer : TYPE object
        DESCRIPTION. the tokenizer object

    """
    token_filter = '! "# $% & () * +, -. / : ; <=>? @ [] ^` {|} ~ “ ” ‘ ’'

    tokenizer = Tokenizer(filters = token_filter) 
    tokenizer.fit_on_texts(list(train))

    # # Rarewords and its Coverage
    count_, totalCount, freq, totalFreq = rare_words_cover(tokenizer, threshold = 4)
    

    tokenizer = Tokenizer(num_words = totalCount - count_, 
                            filters = token_filter) 
    tokenizer.fit_on_texts(list(train))
    
    #convert text sequences into integer sequences
    tr_seq = tokenizer.texts_to_sequences(train) 
    val_seq = tokenizer.texts_to_sequences(validate)
    
    #padding zero upto maximum length
    train = pad_sequences(tr_seq,  maxlen = max_text_length, padding='post')
    validate = pad_sequences(val_seq, maxlen = max_text_length, padding='post')
    
    #size of vocabulary ( +1 for padding token)
    vocab_size = tokenizer.num_words + 1
    return train, validate, vocab_size, tokenizer