import os
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import cv2
import config as cfg



class Source_Data(object):
    
    def __init__(self, batch_times, sequence):
        self.img_size_x = cfg.image_size_x
        self.img_size_y = cfg.image_size_y
        self.g_size_x = cfg.ground_size_x
        self.g_size_y = cfg.ground_size_y
        self.seq = sequence
        self.bsize = cfg.batch_size
        self.need_img_ = cfg.need_img
        self.data_path = os.getcwd()+'/frame-set'
        self.ground_path = os.getcwd()+'/data-set'
        
        self.data_amount = cfg.sample_for_train_
        if (batch_times+1) > self.data_amount:
            self.batch_times = 1
        else:
            self.batch_times = batch_times
            
        self.batch_size = cfg.batch_size
        self.data_path = cfg.data_path
        self.partition = cfg.partition
        
        self.data_cap_map = cfg.area_
        self.random_value4box = cfg.random_val4box
        
    def get_train_data(self): 
        img_name_list = self.seq[self.need_img_*self.batch_times:self.need_img_*(self.batch_times+1)]
        
        img_list = []
        ground_list = []
        score_list = []
        frame_1 = []
        for i in img_name_list:
            img = cv2.imread(data_path + '/' + str(i) + '-5.jpg')
            tmp_img = cv2.imread(data_path + '/' + str(i) + '-4.jpg')
            frame = img - img_tmp
            ground = cv2.imread(ground_path + '/' + str(i) + 'g.jpg')
            #img = data_baggage.get('img'+str(i))
            #ground = data_baggage.get('ground'+str(i))
            #frame1 = data_baggage.get('delta1_frame'+str(i))
        #x 30 pixel random with last 3 areas
            for j in range(len(self.data_cap_map)):
                area = self.data_cap_map[j]
                if j>1:
                    rand_para = random.randint(0,self.random_value4box)
                    area[0] = area[0]+rand_para
                    area[1] = area[1]+rand_para
                img_list.append(img[area[2]:area[3],area[0]:area[1],:])
                ground_list.append(ground[area[2]:area[3],area[0]:area[1]])
                score_list.append(sum(sum(ground[area[2]:area[3],area[0]:area[1]])))
                frame_1.append(frame[area[2]:area[3],area[0]:area[1],:])
        i = 0
        j = 0
        score = sorted(enumerate(score_list), key=lambda x:x[1])
        
        
        train_batch = np.zeros([self.bsize,self.img_size_x,self.img_size_y,3])
        ground_batch = np.zeros([self.bsize,self.g_size_x,self.g_size_y])
        frame1_batch = np.zeros([self.bsize, self.img_size_x, self.img_size_y, 3])
        
        for i in range(len(img_list)):
            if score[i][0]<self.partition[1]:
                train_batch[j,:,:,:] = img_list[i]
                ground_batch[j,:,:] = ground_list[i]
                frame1_batch[j,:,:,:] = frame_1[i]
                j+=1
                if j==self.bsize:
                    break
            if len(score)-score[i][0]<self.partition[0]:
                train_batch[j,:,:,:] = img_list[i]
                ground_batch[j,:,:] = ground_list[i]
                frame1_batch[j,:,:,:] = frame_1[i]
                j+=1
                if j==self.bsize:
                    break
        return train_batch,ground_batch,frame1_batch
        
        
