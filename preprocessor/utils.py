import codecs
import sys
import numpy as np
import datetime
from sklearn import metrics as SKM  



def pad_words(words, max_len):
    delta = max_len - len(words)
    if delta > 0:  
        return np.array(words + [0] * delta)
    else:
        return np.array(words)



def create_input_batch(sentences): 
    batch_size = len(sentences)
    max_length = max([len(s[0]) for s in sentences])   
    l = np.zeros(batch_size) 
    y = np.zeros(batch_size)
    x_ = []
    for k in range(batch_size):
        x_.append(pad_words(sentences[k][0],max_length))
        l[k] = len(sentences[k][0])     
        y[k] = sentences[k][1]
    x = np.stack(x_) 
    return x,l,y

def create_input(sentences, max_len):
    x = np.zeros((len(sentences),max_len))
    y = np.zeros(len(sentences))
    for s, sent in enumerate(sentences):
        for i, word_id in enumerate(sent[0]):
            x[s][i] = word_id
        y[s] = sent[1]
    return x,y

def create_embedding_matrix(id_to_word,embed_model):
    vocab = len(id_to_word)
    size = len(embed_model['dog'])      
    embedding_matrix = np.zeros((vocab, size))
    oov = 0
    drange = np.sqrt(6. / (vocab + size))  

    for i in id_to_word:
        if(id_to_word[i] not in embed_model):  
            embedding_matrix[i] = np.random.uniform(low=-drange, high=drange, size=(size,))  
            oov +=1 
        else:
            embedding_matrix[i] = embed_model[ id_to_word[i] ] 

    print 'Created embedding matrix, OOVs = ',oov
    return embedding_matrix

def metrics(y_test,y_predict): 
    y_predict_b = [ ]
    y_predict_r = [ ]
    assert len(y_test) == len(y_predict)
    for i in range(len(y_predict)):
        y_predict_b += [1 if(y_predict[i]>0.5) else 0]
        y_predict_r += [y_predict[i]]

    acc = SKM.accuracy_score(y_test,y_predict_b)
    pre = SKM.precision_score(y_test,y_predict_b,pos_label=1)
    recall = SKM.recall_score(y_test,y_predict_b,pos_label=1)
    f = SKM.f1_score(y_test,y_predict_b,pos_label=1)
    auc = SKM.roc_auc_score(y_test,y_predict_r)
    return acc,pre,recall,f,auc


