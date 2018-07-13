import tensorflow as tf
import numpy 
import random

import resnet
from data import Source_Data

sample_for_train = 1
sequence = random.sample(range(1,sample_for_train+1),sample_for_train)
Data = Source_Data(1,sequence)
train_batch, ground_batch = Data.get_train_data()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    net = resnet.resnet_v1_50(train_batch)
    print(net)
