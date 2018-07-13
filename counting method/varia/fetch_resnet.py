import tensorflow as tf

class get_resnet(object):
    def __init__(self):
        self.saver = tf.train.import_meta_graph("/home/yuanzhi/figure/mexp/version2/resnet-pretrained/ResNet-L101.meta")
        self.res_graph = tf.get_default_graph()
        with self.res_graph.as_default():
            self.out1 = self.res_graph.get_tensor_by_name("scale1/Relu:0")
            self.out2 = self.res_graph.get_tensor_by_name("scale2/block3/Relu:0")
            self.out3 = self.res_graph.get_tensor_by_name("scale3/block4/Relu:0")
            self.out4 = self.res_graph.get_tensor_by_name("scale4/block4/Relu:0")
            #tf.get_variable_scope().reuse_variables()
            self.images = self.res_graph.get_tensor_by_name("images:0")
            #with tf.Session(graph = self.res_graph) as sess:
            self.sess = tf.Session(graph = self.res_graph)
            self.saver.restore(self.sess,"/home/yuanzhi/figure/mexp/version2/resnet-pretrained/ResNet-L101.ckpt")
            print("----resnet initial done----")


    def get_resnet_output(self, data):
        #with tf.Session(graph = self.res_graph) as sess:
        #sess = tf.Session(graph = self.res_graph)
        print("----get initial output now----")
        out1_, out2_, out3_, out4_ = self.sess.run([self.out1, self.out2, self.out3, self.out4], feed_dict={self.images:data})
        print('out1',out1_.shape)
        print('out2',out2_.shape)
        print('out3',out3_.shape)
        print('out4',out4_.shape)
        return out1_, out2_, out3_

