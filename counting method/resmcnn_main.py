import tensorflow as tf
import numpy as np
import random
import scipy.io as scio
import os
from skimage import io,img_as_ubyte


import cfg
from fetch_resnet import get_resnet
from resmcnn import resnet_mcnn
from open_data import get_data

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

size_x = cfg.train_size_x
size_y = cfg.train_size_y
batch_size = cfg.batch_size
ts_num = cfg.train_times

starting_learning_rate = cfg.starting_learning_rate_
learning_rate_decay = cfg.learning_rate_decay_rate
decay_step_ = cfg.DECAY_STEP

def train():
    data = get_data()
    train_data, ground_data = data.get_train_data(down_size=8)
    loss_file = open(r'resmcnn_train_loss.txt', 'a')
    gpu_options = tf.GPUOptions(allow_growth=True)
    resnet_ = get_resnet()
    data_x1, data_x2, data_x3 = resnet_.get_resnet_output(train_data)
    #raise Exception("whatever")
    
    with tf.Graph().as_default() as g2:
        input_x1 = tf.placeholder(tf.float32, shape=[None, size_x//2, size_y//2, 64])
        input_x2 = tf.placeholder(tf.float32, shape=[None, size_x//4, size_y//4, 256])
        input_x3 = tf.placeholder(tf.float32, shape=[None, size_x//8, size_y//8, 512])
        
        gt = tf.placeholder(tf.float32,shape=[None, size_x//8, size_y//8])
        
        net = resnet_mcnn()
        pred = net.model(input_x1, input_x2, input_x3)
        loss_ = net.loss_layer(pred, gt)
        gloabal_step_ = tf.Variable(0,trainable=False)
        learning_rate = tf.train.inverse_time_decay(starting_learning_rate,global_step=gloabal_step_,decay_steps=decay_step_,decay_rate=learning_rate_decay,staircase=False)
        train_opt = tf.train.AdamOptimizer(learning_rate)
        add_global = gloabal_step_.assign_add(1)
        with tf.control_dependencies([add_global]):
            train_step = train_opt.minimize(loss_)
        saver2 = tf.train.Saver(max_to_keep=20)
    with tf.Session(graph=g2) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(ts_num):
            print('training step:'+str(i)+' / '+str(ts_num)+'\n')
            feed_dict={input_x1:data_x1, input_x2:data_x2, input_x3:data_x3, gt:ground_data}
            _, train_loss, pred_, gt_ = sess.run([train_step, loss_, pred, gt], feed_dict=feed_dict)
            #print(pred_)
            #print('\npred_')
            #print(gt_)
            #print('\ngt_')
            print('train loss: %d' %(train_loss))
            loss_file.write(str(train_loss)+'\n')
            train_data, ground_data = data.get_train_data(down_size=8)
            data_x1, data_x2, data_x3 = resnet_.get_resnet_output(train_data)
            if i % 100 == 0:
                if os.path.isdir('./res_mcnn_save'):
                    saver2.save(sess, './res_mcnn_save/session2.ckpt')
                else:
                    os.mkdir('./res_mcnn_save')
        loss_file.close()





def test():
    data = get_data()
    resnet_ = get_resnet()
    gpu_options = tf.GPUOptions(allow_growth=True)

    for img_num_ in range(1001,1002):
        test_data, test_gronud, test_x, test_y = data.get_test_data(down_size=8, img_num=img_num_)
        data_x1, data_x2, data_x3 = resnet_.get_resnet_output(test_data)

        with tf.Graph().as_default() as g2:
            input_x1 = tf.placeholder(tf.float32, shape=[None, test_x//2, test_y//2, 64])
            input_x2 = tf.placeholder(tf.float32, shape=[None, test_x//4, test_y//4, 256])
            input_x3 = tf.placeholder(tf.float32, shape=[None, test_x//8, test_y//8, 512])

            gt = tf.placeholder(tf.float32,shape=[None, test_x//8, test_y//8])

            net = resnet_mcnn()
            pred = net.model(input_x1, input_x2, input_x3)
            pred = tf.reshape(pred, [test_x//8, test_y//8])
            #saver2 = tf.train.Saver(max_to_keep=20)
        with tf.Session(graph=g2) as sess:
            #sess.run(tf.global_variables_initializer())
            if os.path.exists('./res_mcnn_save/session2.ckpt.index'):
                saver = tf.train.Saver()
                saver.restore(sess, './res_mcnn_save/session.ckpt')
                feed_dict={input_x1:data_x1, input_x2:data_x2, input_x3:data_x3}
                pred_img = sess.run(pred, feed_dict=feed_dict)
                print(type(pred_img))
                tmin, tmax = pred_img.min(), pred_img.max()
                img_show = (pred_img-tmin)/(tmax-tmin)
                img_show = img_as_ubyte(img_show)
                io.imsave('./res_test_mall'+str(img_num_)+'.jpg', img_show)
                print(img_show)
                scio.savemat('./res_test_mall'+str(img_num_), {'anno_img':pred_img})
            else:
                raise Exception("**@liang : ckpt file not found")
            
    




if __name__=='__main__':
    train()
    #test()

