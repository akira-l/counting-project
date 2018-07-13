import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
import skimage.io
from skimage import transform
import numpy as np

import resnet

load_path = "/home/yuanzhi/figure/mexp/version2/resnet-pretrained/"

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, size=300):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = transform.resize(crop_img, (size, size))
    return resized_img

img = load_image("cat.jpg")
batch = img.reshape((1, 300, 300, 3))
#img = cv2.imread("cat.jpg")
#img_ = np.zeros([1,224,224,3])
#img_ = img[0:224,0:224,:]

def get_resnet_output(data):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path+'ResNet-L101.meta')
        saver.restore(sess,"/home/yuanzhi/figure/mexp/version2/resnet-pretrained/ResNet-L101.ckpt")
        graph = tf.get_default_graph()
        out1 = graph.get_tensor_by_name("scale1/Relu:0")
        out2 = graph.get_tensor_by_name("scale2/block3/Relu:0")
        out3 = graph.get_tensor_by_name("scale3/block4/Relu:0")
        images = graph.get_tensor_by_name("images:0")
        #output4 = graph.get_tensor_by_name("scale4/block23/Relu")
        out1_, out2_, out3_ = sess.run([out1, out2, out3], feed_dict={images:data})
        print(output.shape)
    return out1_, out2_, out3_
    '''
    (Pdb) out1.shape
    (1, 150, 150, 64)
    (Pdb) out2.shape
    (1, 75, 75, 256)
    (Pdb) out3.shape
    (1, 38, 38, 512)
    '''


def __init__():
    print('enter')
    get_resnet_output(batch)
    



