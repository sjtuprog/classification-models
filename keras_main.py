from preprocessor.loader import *
from keras_model.kmodel import *
from preprocessor.utils import *
import datetime 
import numpy as np
import sys
import gensim
import random 
np.random.seed(321)

train, dev, test = load_file('corpus/example_data.json') 

dico_words, word_to_id, id_to_word = word_mapping(train+dev+test)

train_data = prepare_dataset(train, word_to_id)
test_data = prepare_dataset(test, word_to_id)
dev_data = prepare_dataset(dev, word_to_id)


max_length = max( [ len(x[0]) for x in (train_data+test_data+dev_data) ]) 
print 'The max_length is:%d' % max_length

x_train, y_train = create_input(train_data, max_length)
x_test, y_test = create_input(test_data, max_length)
x_dev, y_dev = create_input(dev_data, max_length)

print ("%i / %i / %i sentences in train / dev / test." % (len(train_data),  len(dev_data), len(test_data)) )

embedding_model = load_embedding('embedding/google_vector.txt')
embedding_matrix = create_embedding_matrix(id_to_word, embedding_model)
sequence_length = max_length
vector_dim = 300
input_dim = len(id_to_word)
model = KCLF() 
model.build(sequence_length, vector_dim, input_dim, embedding_matrix) 

model.train(x_train, y_train, x_dev, y_dev, 10, epochs=10)

y_predict = model.predict(x_dev)


acc,pre,recall,f,auc = metrics(y_predict, y_test)
print 'Acc:%f, Prec:%f, Reca:%f, F1:%f, AUC:%f' %(acc,pre,recall,f,auc)



