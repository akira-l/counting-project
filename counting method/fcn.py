import tensorflow as tf
import numpy as np
import os
import skimage

import cfg

class fcn_model(object):
    def __init__(self):
        self.test_x = cfg.size_x
        self.test_y = cfg.size_y
        self.train_x = cfg.train_size_x
        self.train_y = cfg.train_size_y
       
       
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
        
    def biases_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
        
    def model(self, x_input):
        with tf.variable_scope('model'):
            W_conv1 = self.weight_variable([9,9,3,36])
            b_conv1 = self.biases_variable([36])
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_input, W_conv1, strides=[1,1,1,1], padding='SAME')+b_conv1)
            
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
            
            W_conv2 = self.weight_variable([7,7,36,72])
            b_conv2 = self.biases_variable([72])
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME')+b_conv2)
            
            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            
            W_conv3 = self.weight_variable([7,7,72,36])
            b_conv3 = self.biases_variable([36])
            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1,1,1,1], padding='SAME')+b_conv3)
            
            W_conv4 = self.weight_variable([7,7,36,24])
            b_conv4 = self.biases_variable([24])
            h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1,1,1,1], padding='SAME')+b_conv4)
            
            W_conv5 = self.weight_variable([7,7,24,16])
            b_conv5 = self.biases_variable([16])
            h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1,1,1,1], padding='SAME')+b_conv5)
            
            W_conv6 = self.weight_variable([1,1,16,1])
            b_conv6 = self.biases_variable([1])
            h_conv6 = tf.nn.relu(tf.nn.conv2d(h_conv5, W_conv6, strides=[1,1,1,1], padding='SAME')+b_conv6)
            
            return h_conv6
            
    def loss_layer(self, pred, gt, stage='train'):
        if stage=='test':
            ground = tf.reshape(gt, [self.test_x//4, self.test_y//4, 1])
        else:
            ground = tf.reshape(gt, [-1, self.train_x//4, self.train_y//4, 1])
        loss = tf.reduce_sum(tf.square(tf.subtract(pred, ground)))
        return loss
        
    
            
            
            
