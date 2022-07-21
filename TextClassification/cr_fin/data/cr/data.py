import csv
import numpy as np
import re
from sklearn.utils import shuffle
import pandas as pd
import pickle

def clean_str(string, TREC=False):
    """
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def read_MR():
    data = []
    vocab = {}
    vocab_em = 1
    sequence_max_length = 0
    with open("rt-polaritydata/pos", "r", encoding="utf-8") as f:
        for line in f:
            x = []
            if line[-1] == "\n":
                line = line[:-1]
            line = clean_str(line)
            sequence = line.split()
            if len(sequence) != 0:
                for i in range(len(sequence)):
                    if sequence[i] not in vocab:
                        vocab.update({sequence[i]:vocab_em})
                        vocab_em += 1
                if sequence_max_length < len(sequence):
                    sequence_max_length = len(sequence)
                x.append(line)
                x.append(1)           
                data.append(x)

    with open("rt-polaritydata/neg", "r", encoding="utf-8") as f:
        for line in f:
            x = []
            if line[-1] == "\n":
                line = line[:-1]
            line = clean_str(line)
            sequence = line.split()
            if len(sequence) != 0:
                for i in range(len(sequence)):
                    if sequence[i] not in vocab:
                        vocab.update({sequence[i]:vocab_em})
                        vocab_em += 1
                if sequence_max_length < len(sequence):
                    sequence_max_length = len(sequence)

                x.append(line)  
                x.append(0)         
                data.append(x)

    data = shuffle(data)
    test_idx = len(data) // 10 * 9

    data_train = pd.DataFrame(data=data[:test_idx])
    data_test = pd.DataFrame(data=data[test_idx:])
    data_train.to_csv("train.csv", sep='\t', index=None,header=None)
    data_test.to_csv("dev.csv", sep='\t', index=None,header=None)
    train_len = test_idx
    text_len = len(data) - train_len
    vocab_len = len(vocab)
    #with open('data/MR/wordmap.json', 'w') as j:
     #   json.dump(vocab, j)
    return vocab_len + 1, sequence_max_length, train_len, text_len

RESULT = read_MR()
print(RESULT)