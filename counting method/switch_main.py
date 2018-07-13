import tensorflow as tf
import numpy as np
import scipy.io as scio
import os

from cpcnn import cpcnn
from data import get_data
import cfg

size_x = cfg.size_x
size_y = cfg.size_y
batch_size = cfg.batch_size
ts_num = 5


def train():
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session(graph=switch_graph) as sess:
        
        xs = tf.placeholder(tf.float32, shape=[None, 300, 300, 3])
        
        
        model = cpcnn()
        model.model(xs)
        feed_dict={xs:train_data}
        switch_signal = sess.run(model.switch, feed_dict=feed_dict)
        signal = switch_signal.index(max(switch_signal))+1
        if signal == 1:
            with tf.Session(graph=sub_graph1) as sess:
                pred = model.result1
                gt = tf.placeholder(tf.float32, shape=[None, 75, 75])
                loss = model.loss_layer(pred, gt)
        if signal == 2:
            with tf.Session(graph=sub_graph2) as sess:
        
        loss = model.loss_layer(pred, gt)
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
        
        data = get_data()
        
        sess.run(tf.global_variables_initializer())
        for i in range(1, ts_num):
            train_data, ground_data = data.get_train_data(down_size=4)
            feed_dict={xs:train_data, gt:ground_data, model.output_shape_1:[batch_size, 150, 150, 32], model.output_shape_2:[batch_size, 300, 300, 16]}
            _, train_loss = sess.run([train_step, loss], feed_dict=feed_dict)
            print("loss:",train_loss)
            
            
def test():
    pass
        
        
if __name__ == '__main__':
    train()
    #test()
    
    
