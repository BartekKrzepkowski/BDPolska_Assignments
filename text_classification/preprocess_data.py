import re
import numpy as np
from functools import reduce
from string import digits, punctuation
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def get_char_decoder(data):
    unique_in_rows = data.apply(lambda x: np.unique(list(x)))
    unique_chars = reduce(lambda x,y: np.unique(np.concatenate([x,y])), unique_in_rows)
    int2char = dict(enumerate(unique_chars))
    int2char = dict(((k+1, v) for k, v in int2char.items()))
    int2char[0] = "<PAD>"
    return int2char


def get_char_encoder(int2char):
    return dict(((v,k) for k,v in int2char.items()))


def get_encoded_char_input(data, char2int, max_len_seq):
    data = [[char2int[x] for x in row] for row in data]
    encoded_data = pad_sequences(data, value=0, padding="post", maxlen=max_len_seq)
    return np.array(encoded_data)


def splitter_1(data):
    train_df = data[data.df_ == 0]
    val_df = data[data.df_ == 1]
    test_df = data[data.df_ == 2]
    return train_df, val_df, test_df


def splitter_2(data):
    size = data.shape[0]
    train = int(0.7*size)
    val = int(0.15*size)
    idx = np.random.permutation(np.arange(size))
    idx_train = idx[: train]
    idx_val = idx[train: val+train]
    idx_test = idx[train+val: ]
    return data.iloc[idx_train], data.iloc[idx_val], data.iloc[idx_test]


def preprocess_data_char(data_org, x_label, y_label, max_len_seq=None):
    data = data_org.copy()
    sent = data[x_label]
    int2char = get_char_decoder(sent)
    char2int = get_char_encoder(int2char)
    if not max_len_seq:
        max_len_seq = sent.str.len().max()
    
    train_df, val_df, test_df = splitter_2(data)
    
    x_train = get_encoded_char_input(train_df[x_label], char2int, max_len_seq)
    y_train = train_df[y_label].values
    x_val = get_encoded_char_input(val_df[x_label], char2int, max_len_seq)
    y_val = val_df[y_label].values
    x_test = get_encoded_char_input(test_df[x_label], char2int, max_len_seq)
    y_test = test_df[y_label].values

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), int2char



def preprocess_data_word(data_org, x_label, y_label, max_len_seq=None):
    data = data_org.copy()
    if max_len_seq:
        review_cleaned= data[x_label].str.split().apply(lambda x: " ".join(x[: max_len_seq]))
        review_cleaned = review_cleaned.apply(lambda x: re.sub(r' +', ' ', x))
        data[x_label] = review_cleaned.apply(lambda x: re.sub(r' (?=[\.,!?&\-])','', x))

        
    train_df, val_df, test_df = splitter_2(data)
    
    x_train = train_df[x_label].values[:, np.newaxis]
    y_train = train_df[y_label].values
    x_val = val_df[x_label].values[:, np.newaxis]
    y_val = val_df[y_label].values
    x_test = test_df[x_label].values[:, np.newaxis]
    y_test = test_df[y_label].values
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), data

def preprocess_data_word_tokenizer(data_org, x_label, y_label, max_len_seq=None):
    data = data_org.copy()
    if max_len_seq:
        review_cleaned= data[x_label].str.split().apply(lambda x: " ".join(x[: max_len_seq]))
        review_cleaned = review_cleaned.apply(lambda x: re.sub(r' +', ' ', x))
        data[x_label] = review_cleaned.apply(lambda x: re.sub(r' (?=[\.,!?&\-])','', x))
        
    tok = Tokenizer()
    tok.fit_on_texts(data[x_label].values)
    encoded_data = tok.texts_to_sequences(data[x_label].values)
    max_len_seq = min(max_len_seq, max([len(row) for row in encoded_data])) if max_len_seq else max([len(row) for row in encoded_data])
    encoded_data = pad_sequences(encoded_data, value=0, padding="post", maxlen=max_len_seq)
    encoded_data = np.array(encoded_data)
    tok.index_word[0] = "<PAD>"
    
    x_train = encoded_data[data.df_ == 0]
    y_train = data[data.df_ == 0][y_label].values[:, np.newaxis]
    x_val = encoded_data[data.df_ == 1]
    y_val = data[data.df_ == 1][y_label].values[:, np.newaxis]
    x_test = encoded_data[data.df_ == 2]
    y_test = data[data.df_ == 2][y_label].values[:, np.newaxis]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), tok.index_word


def clear_text(text, is_all_lower=True):
    punct = re.sub(r'[\.,!?&\-]', '', punctuation)
    punctuation_table = str.maketrans({key: "#" for key in punct})
    for char in ["\"", "\'"]:
        del punctuation_table[ord(char)]
    
    review_cleaned = text.apply(lambda x: re.sub(r'[^\x00-\x7F]', ' ', x))
    review_cleaned = review_cleaned.apply(lambda x: re.sub(r'[0-9]', '9', x))
    review_cleaned = review_cleaned.apply(lambda x: x.translate(punctuation_table))
    review_cleaned = review_cleaned.apply(lambda x: re.sub(r' +', ' ', x))
    review_cleaned = review_cleaned.apply(lambda x: re.sub(r' (?=[\.,!?&\-])','', x))
    
    if is_all_lower:
        review_cleaned = review_cleaned.str.lower()
        
    return review_cleaned
