import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import sys
sys.path.append('./data/')
from data import getty

def sample(c, h, l, W, U, V, bh, by, char2idx, idx2char):
    cs = []
    cs.append(c)
    x = np.zeros([1, 27])
    x[0][char2idx[c]] = 1.0
    for i in range(l):
        h = np.tanh(x @ U + h @ W + bh)
        y_next = h @ V + by
        i_next = np.argmax(y_next)
        c_next = idx2char[i_next]
        x = np.zeros_like(x)
        x[0][i_next] = 1.0
        cs.append(c_next)
    return ''.join(cs)

def main():
    l = 5
    d_in, d_hidden, d_out = 27, 100, 27
    
    x = tf.placeholder(tf.float32, [l, 1, d_in], name='x')
    y = tf.placeholder(tf.int32,   [l, 1], name='y')
    h = tf.placeholder(tf.float32, [1, d_hidden], name='h')
    
    W = tf.Variable(np.random.randn(d_hidden, d_hidden), dtype=tf.float32, name='W')
    U = tf.Variable(np.random.randn(d_in, d_hidden),     dtype=tf.float32, name='U')
    V = tf.Variable(np.random.randn(d_hidden, d_out),    dtype=tf.float32, name='V')
    bh = tf.Variable(np.zeros([1, d_hidden]),            dtype=tf.float32)
    by = tf.Variable(np.zeros([1, d_out]),               dtype=tf.float32)

    
    x_seq = tf.unstack(x)
    y_seq = tf.unstack(y)
    yhat_seq = []
    h_seq = []
    h_seq.append(h)
    for xi in x_seq:
        h_next = tf.tanh(tf.matmul(xi, U) + tf.matmul(h_seq[-1], W) + bh)
        yhat = tf.matmul(h, V) + by
        yhat_seq.append(yhat)
        h_seq.append(h_next)
        
    losses = [tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(yi, d_out), logits=yhati) for yhati, yi in zip(yhat_seq, y_seq)]
    loss = tf.reduce_mean(losses)
    train_op = tf.train.AdagradOptimizer(1).minimize(loss)
    

    init = tf.global_variables_initializer()
    
    
    text, x_train, y_train, char2idx, idx2char = getty()
    x_ = x_train[:l,]
    y_ = y_train[:l]
    h_ = np.zeros([1, d_hidden], dtype=np.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1000):
            _, l, V_, W_, U_, bh_, by_, h_, y_seq_ = sess.run([train_op, loss, V, W, U, bh, by, h, y_seq], feed_dict={x:x_, y:y_, h:h_})
            if i % 1000 == 0:
                txt = sample('f', h_, 20, W_, U_, V_, bh_, by_, char2idx, idx2char)
                print(i, l, txt)
                #print(y_seq)
        
        file_writer = tf.summary.FileWriter('./tf/logs')


        


if __name__ == '__main__':
    main()

