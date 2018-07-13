import tensorflow as tf
import numpy as np
import scipy.io as scio
import os
from skimage import io,img_as_ubyte

from cpcnn import cpcnn
from data import get_data
import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

size_x = cfg.size_x
size_y = cfg.size_y
batch_size = cfg.batch_size
epoch = 40
ts_num = epoch*(256//batch_size)


def train():
    loss_file = open(r'cpcnn_loss.txt','a')
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
        xs = tf.placeholder(tf.float32, shape=[None, 300, 300, 3])
        #ys = tf.placeholder(tf.float32, shape=[None, 75, 75])
        gt = tf.placeholder(tf.float32, shape=[None, 300, 300])
        model = cpcnn()
        pred, disc = model.model(xs)
        
        loss = model.loss_layer(pred, gt, disc)
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
        
        data = get_data()
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=20)
        for i in range(1, ts_num):
            train_data, ground_data = data.get_train_data()
            feed_dict={xs:train_data, gt:ground_data, model.output_shape_1:[batch_size, 150, 150, 32], model.output_shape_2:[batch_size, 300, 300, 16]}
            _, train_loss = sess.run([train_step, loss], feed_dict=feed_dict)
            print("loss:",train_loss)
            loss_file.write(str(train_loss)+'\n')
            if i % 100 == 0:
                if os.path.isdir('./cpcnn_save'):
                    saver.save(sess, './cpcnn_save/session.ckpt')
                else:
                    os.mkdir('./cpcnn_save')
        loss_file.close()
            
            
def test():
    data = get_data()
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        xs = tf.placeholder(tf.float32, shape=[300, 300, 3])
        xs_ = tf.reshape(xs, [-1, 300, 300, 3])
        gt = tf.placeholder(tf.float32, shape=[300, 300])
        gt_ = tf.reshape(gt, [-1, 300, 300])
        model = cpcnn(stage='test')
        pred, disc = model.model(xs_)
        print('pred shape:', pred.get_shape().as_list())
        pred_show = tf.reshape(pred, [300,300])
        #sess.run(tf.global_variables_initializer())
        #saver = tf.train.Saver(var_list = tf.trainable_variables())
        if os.path.exists('./cpcnn_save/session.ckpt.index'):
            saver = tf.train.Saver()
            saver.restore(sess, './cpcnn_save/session.ckpt')
            test_data, test_ground = data.get_test_data(down_size=1, method='cpcnn')
            feed_dict={xs:test_data, gt:test_ground, model.output_shape_1:[1, 150, 150, 32], model.output_shape_2:[1, 300, 300, 16]}
            pred_img = sess.run(pred_show, feed_dict=feed_dict)
            #print("loss %f" %test_loss)
            tmin, tmax = pred_img.min(), pred_img.max()
            img_show = (pred_img-tmin)/(tmax-tmin)
            img_show = img_as_ubyte(img_show)
            io.imsave('./cpcnn_test.jpg',img_show)
            print(img_show)
        else:
            raise Exception('**@liang==== ckpt file not found')
        
        
if __name__ == '__main__':
    #train()
    test()
    
    
