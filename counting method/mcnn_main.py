import tensorflow as tf
import numpy as np
import scipy.io as scio
import os
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import plot, savefig
from skimage import io,data,img_as_ubyte
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from mcnn import mcnn_model
from data import get_data
import cfg

size_x = cfg.size_x
size_y = cfg.size_y
batch_size = cfg.batch_size
ts_num = cfg.train_times


def train():
    loss_file = open(r'ncnn_train_loss.txt', 'a')
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
        xs = tf.placeholder(tf.float32, shape=[None, 300, 300, 3])
        #ys = tf.placeholder(tf.float32, shape=[None, 75, 75])
        gt = tf.placeholder(tf.float32, shape=[None, 75, 75])
        model = mcnn_model()
        pred = model.model(xs)
        
        loss = model.loss_layer(pred, gt)
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
        
        data = get_data()
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=20)
        for i in range(1, ts_num):
            train_data, ground_data = data.get_train_data(down_size=4)
            _, train_loss = sess.run([train_step, loss], feed_dict={xs:train_data, gt:ground_data})
            print("loss:",train_loss)
            loss_file.write(str(train_loss)+'\n')
            if i % 100 == 0:
                if os.path.isdir('./mcnn_save'):
                    saver.save(sess, './mcnn_save/session2.ckpt')
                else:
                    os.mkdir('./mcnn_save')
        loss_file.close()

                
            
            
def test():
    loss_file = open(r'mcnn_test_loss.txt','a')
    data = get_data()
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        xs = tf.placeholder(tf.float32, shape=[720, 1280, 3])
        xs_ = tf.reshape(xs, [-1, 720, 1280, 3])
        gt = tf.placeholder(tf.float32, shape=[180, 320])
        gt_ = tf.reshape(gt, [-1, 180,320])
        model = mcnn_model()
        pred = model.model(xs_)
        pred_show = tf.reshape(pred, [180,320])
        loss = model.loss_layer(pred, gt_, stage='test')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list = tf.trainable_variables())
        if os.path.exists('./mcnn_save/session2.ckpt.index'):
            saver.restore(sess, './mcnn_save/session2.ckpt')
            img, ground = data.get_test_data(down_size=4)
            pred_img, test_loss = sess.run([pred_show, loss], feed_dict={xs:img, gt:ground})
            loss_file.write(str(test_loss)+'\n')
            tmin, tmax = pred_img.min(), pred_img.max()
            img_show = 255*(pred_img-tmin)/(tmax-tmin)
            img_show = img_as_ubyte(img_show)
            io.imsave('./mcnn_test.jpg',img_show)
            print(img_show)
        else:
            raise Exception('**@liang==== ckpt file not found')
    loss_file.close()
            


if __name__ == '__main__':
    #train()
    test()
    
    

