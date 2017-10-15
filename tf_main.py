from preprocessor.loader import *
from tf_model.model import *
from preprocessor.utils import *
import datetime
from collections import OrderedDict
import numpy as np
import sys
import gensim
import random
from tqdm import tqdm
np.random.seed(321)

train, dev, test = load_file('corpus/example_data.json') 

dico_words, word_to_id, id_to_word = word_mapping(train+dev+test)

train_data = prepare_dataset(train, word_to_id)
test_data = prepare_dataset(test, word_to_id)
dev_data = prepare_dataset(dev, word_to_id)


print ("%i / %i / %i sentences in train / dev / test." % (len(train_data),  len(dev_data), len(test_data)) )

embedding_model = load_embedding('embedding/google_vector.txt') 
embedding_matrix = create_embedding_matrix(id_to_word, embedding_model)

parameters = OrderedDict() 
parameters['vocab_dim'] = len(word_to_id) 
parameters['vector_dim'] = 300
parameters['hidden_dim'] = 100
parameters['lr_rate'] = 0.001    
parameters['embedding_matrix'] =  embedding_matrix
parameters['train_embed'] = False

with tf.device("/cpu:0"):
	model = CLF(**parameters)
print("Building model finished") 

n_epochs = 10
batch_size = 10
count = 0

best_f1 = 0.0
best_ep = -1

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        print ("Starting epoch %i..." % epoch)
        start_time_epoch = datetime.datetime.now()
        random.shuffle(train_data) 
        epoch_costs = []  
        for i in tqdm(range(len(train_data)/batch_size)):
            start_time = datetime.datetime.now() 
            batch_data = train_data[i*batch_size: (i+1)*batch_size]
            if len(batch_data) < batch_size:
                batch_data.extend(train_data[:batch_size - len(batch_data)])  
            x,l,y = create_input_batch(batch_data)  
            new_loss = model.train(sess,x,l,y)
            epoch_costs.append(new_loss)  

        x,l,y_true = create_input_batch(dev_data)    
        y_pred = model.predict(sess,x,l)
        a,p,r,f1,auc = metrics(y_true, y_pred) 
        print 'Prec: %0.3f, Reca: %0.3f, F1: %0.3f' %(p,r,f1)
        if f1 > best_f1:
        	best_f1 = f1
        	best_ep = epoch 
        end_time_epoch = datetime.datetime.now()
        cost_time_epoch = (end_time_epoch - start_time_epoch).seconds
        print ("Epoch %i done. Average cost: %f, Cost time: %i sec\n" % (epoch, np.mean(epoch_costs), cost_time_epoch))
    print('Best F1: %f, at best Epoch: %d' %(best_f1, best_ep))










