import numpy as np
import random
np.random.seed(1008)
import tensorflow as tf
tf.set_random_seed(1234)
import json
import keras  
from keras.preprocessing import sequence
from keras.utils import np_utils 
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D, LSTM
from keras import metrics 
from keras import backend as K
K.set_learning_phase(1)
from keras.regularizers import l2
from keras.callbacks import *
# from visualizer import *
from keras.models import * 
from keras.layers.core import *
from keras.layers import Input, Embedding, LSTM, Dense, merge, TimeDistributed ,Bidirectional

class KCLF(object):
    def __init__(self):
        self.path = './models/'

    def build(self, sequence_length, vector_dim, input_dim, embedding_matrix):
        main_input = Input(shape=(sequence_length,)) 

        embed_layer = Embedding(input_dim, vector_dim, weights=[embedding_matrix], trainable=False)(main_input)

        bilstm = Bidirectional(LSTM(100,return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(embed_layer) 
        #sum_dim1 = Lambda(lambda xin: K.sum(xin, axis=1))(main_input) 
        out = Dense(1, activation='sigmoid')(bilstm)
        self.model = Model(input=main_input, output= out) 
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def train(self,x_train, y_train, x_test, y_test, batch_size, epochs):

        self.model.fit(x_train,y_train,batch_size=batch_size,validation_data=(x_test,y_test),epochs=epochs)

    def predict(self, x):
        return self.model.predict(x)


        