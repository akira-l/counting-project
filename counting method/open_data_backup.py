import numpy as np
import os
import scipy.io as scio
import random
import skimage.io
import skimage.transform
import cfg

class get_data(object):
    def __init__(self):
        self.size_x = cfg.train_size_x
        self.size_y = cfg.train_size_y
        self.train_data_path = "./crowd dataset/ShanghaiTech/part_B/train_data/images/"
        self.ground_data_path = "./crowd dataset/ShanghaiTech/part_B/train ground B/"

        self.test_data_path = "./crowd dataset/ShanghaiTech/part_B/test_data/images/"
        self.test_ground_path = "./crowd dataset/ShanghaiTech/part_B/test ground B/"
        self.total_data_num = cfg.total_data_num
        self.batch_size = cfg.batch_size
        
    def get_train_data(self, down_size=1):
        batch_list = random.sample(range(1,self.total_data_num+1),self.batch_size)
        train_data = np.zeros([self.batch_size,self.size_x, self.size_y,3])
        ground_data = np.zeros([self.batch_size,self.size_x//down_size,self.size_y//down_size])
        
        name_num = 0
        
        for img_num in batch_list:
            img = skimage.io.imread(self.train_data_path+'IMG_'+str(img_num)+'.jpg')
            width = img.shape[0]
            height = img.shape[1]
            #print(self.train_data_path+'IMG_'+str(img_num)+'.jpg')
            rand_area_x = random.randint(0,width-self.size_x)
            rand_area_y = random.randint(0,height-self.size_y)
            train_data[name_num,:,:,:] = img[rand_area_x:rand_area_x+self.size_x,rand_area_y:rand_area_y+self.size_y,:]
            ground = scio.loadmat(self.ground_data_path+'ground'+str(img_num)+'.mat')
            #ground = scio.loadmat(self.ground_data_path+'465.mat')
            ground = ground.get('anno_img')
            
            ground = 10000*ground[rand_area_x:rand_area_x+self.size_x,rand_area_y:rand_area_y+self.size_y]
            if down_size != 1:
                ground = skimage.transform.resize(ground, (self.size_x//down_size,self.size_y//down_size))
            ground_data[name_num,:,:] = ground
        return train_data, ground_data
            

    def get_test_data(self, down_size=1, img_num=1):
        #img_num = 1 # for fcn/cpcnn/mcnn etc.->256
        img = skimage.io.imread(self.test_data_path+"IMG_"+str(img_num)+'.jpg')
        test_x = img.shape[0]
        test_y = img.shape[1]
        test_data = np.zeros([1, test_x, test_y, 3])
        test_ground = np.zeros([1, test_x//down_size, test_y//down_size])
        test_data[0,:,:,:] = img
        ground = scio.loadmat(self.test_ground_path+"ground"+str(img_num)+'.mat')
        ground = ground.get('anno_img')
        if down_size != 1:
            ground = skimage.transform.resize(ground, (test_x//down_size, test_y//down_size))
        test_ground[0,:,:] = ground
        return test_data, test_ground, test_x, test_y 
            
       
