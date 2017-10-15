from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np
 


def bi_lstm(input_, seq_len, h_dim, name='n'): 
    with tf.variable_scope("lstm"+name):
        lstm_fw_cell = rnn.LSTMCell(h_dim)
        lstm_bw_cell = rnn.LSTMCell(h_dim) 
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn( lstm_fw_cell,  lstm_bw_cell,
                     input_,  sequence_length=seq_len, dtype=tf.float32)
        return  tf.concat([output_fw, output_bw], axis=-1, name='bi-lstm'+name) 

