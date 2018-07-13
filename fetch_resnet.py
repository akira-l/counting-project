import tensorflow as tf
import os

import config as cfg

class get_resnet(object):
    def __init__(self):
        self.saver = tf.train.import_meta_graph(os.getcwd()+"/resnet-pretrained/ResNet-L50.meta")
        self.res_graph = tf.get_default_graph()
        self.para_save_path = cfg.para_save_path_
        with self.res_graph.as_default():
            self.out1 = self.res_graph.get_tensor_by_name("scale1/Relu:0")
            self.out2 = self.res_graph.get_tensor_by_name("scale2/block3/Relu:0")
            self.out3 = self.res_graph.get_tensor_by_name("scale3/block4/Relu:0")
            #tf.get_variable_scope().reuse_variables()
            self.images = self.res_graph.get_tensor_by_name("images:0")
            #with tf.Session(graph = self.res_graph) as sess:
            self.sess = tf.Session(graph = self.res_graph)
            self.saver.restore(self.sess,os.getcwd()+"/resnet-pretrained/ResNet-L50.ckpt")
            self.saver_res = tf.train.Saver(max_to_keep=30)

    def get_resnet_output(self, data):
        #with tf.Session(graph = self.res_graph) as sess:
        #sess = tf.Session(graph = self.res_graph)
        out1_, out2_, out3_ = self.sess.run([self.out1, self.out2, self.out3], feed_dict={self.images:data})
        return out1_, out2_, out3_

    def save_resnet(self, name):
        self.saver_res.save(self.sess,name)


