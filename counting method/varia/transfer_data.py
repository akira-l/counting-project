import numpy as np
import scipy.io as scio
import os

import config as cfg


class get_data(object):
    
    def __init__(self,train_flag=False,test_flag=False):
        self.train_flag = train_flag
        self.test_flag = test_flag
        self.area_para = [[25,185,271,471],[121,281,138,338],[121,281,272,472],[279,439,71,271],[279,439,271,471],[441,601,7,207],[442,602,207,407],[560,720,2,202],[560,720,169,369],[35,195,471,671],[35,195,671,871],[158,318,471,671],[158,318,611,811],[158,318,811,1011],[279,439,471,671],[279,439,625,825],[279,439,805,1005],[279,439,987,1187],[439,599,437,637],[439,599,637,837],[439,599,707,907],[439,599,907,1107],[439,599,1080,1280],[560,720,369,569],[560,720,501,701],[560,720,701,901],[560,720,880,1080],[560,720,1080,1280]]
        self.data()
    def data(self):
        
        fetch_data = scio.loadmat('addition.mat')
        if self.train_flag is True:
            fetch_img_mount = 10
        if self.test_flag is True:
            fetch_img_number = 201
            train_batch = np.zeros([280,3,160,200])
            ground_batch = np.zeros([280,160,200])
            img = fetch_data.get('img'+str(fetch_img_number))
            ground = fetch_data.get('ground'+str(fetch_img_number))
            for k in range(28):
                area = self.area_para[k]
                train_batch[k,0,:,:] = img[area[0]:area[1],area[2]:area[3],0]
                train_batch[k,1,:,:] = img[area[0]:area[1],area[2]:area[3],1]
                train_batch[k,2,:,:] = img[area[0]:area[1],area[2]:area[3],2]
                ground_batch[k,:,:] = ground[area[0]:area[1],area[2]:area[3]]
            return train_batch,ground_batch
            
        train_batch = np.zeros([280,3,160,200])
        ground_batch = np.zeros([280,160,200])
        counter = 0
        for i in range(1,fetch_img_mount+1):
            img = fetch_data.get('img'+str(i))
            ground = fetch_data.get('ground'+str(i))
            for j in range(28):
                area = self.area_para[j]
                train_batch[counter,0,:,:] = img[area[0]:area[1],area[2]:area[3],0]
                train_batch[counter,1,:,:] = img[area[0]:area[1],area[2]:area[3],1]
                train_batch[counter,2,:,:] = img[area[0]:area[1],area[2]:area[3],2]
                ground_batch[counter,:,:] = ground[area[0]:area[1],area[2]:area[3]]
                counter += 1
        return train_batch,ground_batch
        


                
