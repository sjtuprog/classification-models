from tensorflow.contrib import rnn
import tensorflow as tf
from nn import * 
import datetime
import numpy as np

class CLF(object):
    def __init__(self, vocab_dim, vector_dim, hidden_dim, lr_rate, embedding_matrix, train_embed):
        self.path = './models/'
        

        self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
        self.l = tf.placeholder(tf.int32, [None], name='len_x1')
        self.input_y = tf.placeholder(tf.float32, [None], name='input_y')    
        self.dropout = tf.placeholder(tf.float32, name="dropout")
        self.embeddings =  tf.get_variable('embeddings', (vocab_dim, vector_dim),
            tf.float32, initializer=tf.constant_initializer(embedding_matrix), trainable=train_embed)

        x = tf.nn.embedding_lookup(self.embeddings, self.input_x) 
        x  = tf.nn.dropout(x, keep_prob= 1 - self.dropout, name='dropout') 
        
        
        
        last_x1 = bi_lstm(x,None,hidden_dim,'xx') 

        last_x1 = last_x1[:,-1,:]

        self.y = tf.layers.dense(last_x1, 1) 
        self.y = tf.squeeze(self.y)  
        self.output_y = tf.nn.sigmoid(self.y)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.y)
        self.loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.AdamOptimizer(lr_rate)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
         
        


    def train(self, sess, x,l,y):
        
        feed_dict_ = {}
        feed_dict_[self.input_x] = x 
        feed_dict_[self.input_y] = y
        feed_dict_[self.l] = l
        feed_dict_[self.dropout] = 0.0
        _, _ ,new_loss = sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict_)
        return new_loss



    def predict(self, sess, x,l):   
        y_preds = []
        feed_dict_ = {}
        feed_dict_[self.input_x] = x
        feed_dict_[self.l] = l
        feed_dict_[self.dropout] = 0.0
        y_pred = sess.run(self.output_y, feed_dict=feed_dict_)   
        return y_pred 
        