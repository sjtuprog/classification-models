from preprocessor.loader import *
import numpy as np
from preprocessor.utils import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
import os


def data_to_vector(data):
	x_word, y = data
	x = np.zeros(len(word_to_id)) 
	for w in x_word:
		x[w]+=1  
	return x, y

train, dev, test = load_file('corpus/example_data.json') 

dico_words, word_to_id, id_to_word = word_mapping(train) 

train_data = prepare_dataset(train, word_to_id)
test_data = prepare_dataset(test, word_to_id)


x_train = []
x_test = []

y_train = []
y_test = []

for t in train_data:
	v, y = data_to_vector(t)
	x_train.append(v)
	y_train.append(y)

for t in test_data:
	v, y = data_to_vector(t)
	x_test.append(v)
	y_test.append(y) 



clf = LogisticRegression(C=1.0, dual=False, fit_intercept=True, intercept_scaling=1, class_weight='balanced',penalty='l2',n_jobs=4)


clf.fit(x_train,y_train)

y_predict = clf.predict(x_test)

a,p,r,f,auc = metrics(y_test, y_predict)

print 'Acc:%f, Prec:%f, Reca:%f, F1:%f, AUC:%f' %(a,p,r,f,auc)

