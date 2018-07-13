import os
import scipy.io as scio
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import config as cfg

class Source_Data(object):
    
    def __init__(self, batch_times, sequence):
        self.img_size_x = cfg.image_size_x
        self.img_size_y = cfg.image_size_y
        self.g_size_x = cfg.ground_size_x
        self.g_size_y = cfg.ground_size_y
        self.seq = sequence
        #print self.seq
        self.data_amount = cfg.sample_for_train
        if (batch_times+1) > self.data_amount:
            self.batch_times = 1
        else:
            self.batch_times = batch_times
            
        self.batch_size = cfg.batch_size
        self.data_path = cfg.data_path
        self.partition = cfg.partition
        self.area_para = [[25,185,271,471],[121,281,138,338],[121,281,272,472],[279,439,71,271],[279,439,271,471],[441,601,7,207],[442,602,207,407],[560,720,2,202],[560,720,169,369],[35,195,471,671],[35,195,671,871],[158,318,471,671],[158,318,611,811],[158,318,811,1011],[279,439,471,671],[279,439,625,825],[279,439,805,1005],[279,439,987,1187],[439,599,437,637],[439,599,637,837],[439,599,707,907],[439,599,907,1107],[439,599,1080,1280],[560,720,369,569],[560,720,501,701],[560,720,701,901],[560,720,880,1080],[560,720,1080,1280]]
        
        #self.train_data, self.ground_data = self.get_train_data(self.seq)
        
        
        
    def get_train_data(self):
        data_num = self.seq[self.batch_times]
        data_bag = scio.loadmat(self.data_path+'data'+str(data_num)+'.mat')
        img = data_bag.get('img'+str(data_num))
        ground = np.double(data_bag.get('ground'+str(data_num)))
        area_score = data_bag.get('score'+str(data_num))
        pos, neg = self.get_train_target(area_score)
        read_seq = pos+neg
        read_seq = random.sample(read_seq,len(read_seq))
        train_batch = []
        ground_batch = []
        for i in range(len(read_seq)):
            area = self.area_para[read_seq[i][0]]
            train_batch.append(img[area[0]:area[1],area[2]:area[3],:])
            '''
            plt.figure("img test")
            plt.imshow(img[area[0]:area[1],area[2]:area[3],:])
            plt.show()
            '''
            
            ground_save = ground[area[0]:area[1],area[2]:area[3]]
            '''
            plt.figure("ground testf")
            plt.imshow(ground_save)
            plt.show()
            '''
            #ground = scipy.ndimage.zoom(ground_save,0.5)
            #print 'down ground', ground_save
            ground_batch.append(ground_save)
        return train_batch,ground_batch
    
    
    
    def get_train_target(self,score):
        negtive_sample = []
        positive_sample = []
        score_list = np.argsort(score)
        for i in range(self.partition[1]):
            negtive_sample.append([score_list[0][i]])
        for j in range(self.partition[0]):
            positive_sample.append([score_list[0][27-j]])
        return positive_sample,negtive_sample



