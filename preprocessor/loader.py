import codecs
import sys
import numpy as np
import datetime
import json
import re
import time
import random

def load_file(fin):
    train = []
    test = []
    dev = []
    with codecs.open(fin,'r','utf8') as f:
        for line in f:
            x = json.loads(line)
            x['text'] = clean_str(x['text'])
            if x['split']=='train':
                train.append(x)
            elif x['split']=='test':
                test.append(x)
            else:
                dev.append(x)
    return train, dev, test


def clean_str(string):  
    string = re.sub(r"['\"]+","",string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string) 
    string = re.sub(r'``','', string)
    return string.strip().lower()

def create_dico(item_list): 
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico):
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item    

def word_mapping(sentences): 
    words = [  s['text'].split() for s in sentences] 
    dico = create_dico(words)
    dico['<UNK>'] = sys.maxsize
    word_to_id, id_to_word = create_mapping(dico)
    print ("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word

def prepare_dataset(sentences, word_to_id): 
    data = []
    for s in sentences:  
        x =  [word_to_id[w if w in word_to_id else '<UNK>'] for w in s['text'].split()] 
        y = s['label'] 
        data.append([x,y])
    return data

def load_embedding(gloveFile):
    print "Loading Glove Model"
    start_time = time.clock()
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.asarray([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print "Done.",len(model)," words loaded!"

    end_time = time.clock()
    print 'Time in read embedding %f s' %(end_time-start_time)
    return model

