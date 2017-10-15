from loader import *
from model import *
from utils import *
import datetime
from collections import OrderedDict
import numpy as np
import sys
import gensim
np.random.seed(321)
import time


def loadGloveModel(gloveFile):
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
    print 'Time in read %f s' %(end_time-start_time)
    return model

def loadGoogleModel(googleFile):
    print "Loading Google word2vec Model"
    start_time = time.clock()
    model = gensim.models.KeyedVectors.load_word2vec_format(googleFile, unicode_errors='ignore',binary=True)  

    end_time = time.clock()
    print 'Time in read %f s' %(end_time-start_time)
    return model 


train, dev, test = load_files('corpus/fox_dev.json') 

dico_words, word_to_id, id_to_word = word_mapping(train)

google_path = '/Users/leigao/Documents/NLP/pre-trained-wordvec/GoogleNews-vectors-negative300.bin'
glove_300 = '/Users/leigao/Documents/NLP/pre-trained-wordvec/glove.840B.300d.txt'


gmodel = loadGoogleModel(google_path)
i = 0
f = open('google_vector_fox.txt','w')
for w in word_to_id:
    if w in gmodel:
        x = w +' '+ ' '.join(np.array_str(gmodel[w])[1:-1].split())+'\n'
        f.writelines(x)
        i+=1
print 'finally %d' % i





