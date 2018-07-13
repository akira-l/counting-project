import os
import scipy.io as scio
import random
import numpy as np
import config as cfg


class Source_Data(object):

    def __init__(self, batch_times, sequence):
        self.img_size_x = cfg.image_size_x
        self.img_size_y = cfg.image_size_y
        self.g_size_x = cfg.ground_size_x
        self.g_size_y = cfg.ground_size_y
        self.seq = sequence
        self.bsize = cfg.batch_size
        self.need_amount = cfg.need_amount

        self.data_amount = cfg.sample_for_train_
        if (batch_times+1) > self.data_amount:
            self.batch_times = 1
        else:
            self.batch_times = batch_times

        self.batch_size = cfg.batch_size
        self.data_path = cfg.data_path
        self.partition = cfg.partition

    '''
    capture two areas in every frame images
    one in the rear, one in front 1-400 320-720 
    in horizontal: totally random select
    total fetch: [batch_size*1.5] select the maxinal XXX and minimal XXX into a batch
    randomly symitric turn image area. 
    '''

    def get_train_data(self):
        name_list = self.seq[self.need_amount*self.batch_times:self.need_amount*(self.batch_times+1)]

        img_list = []
        ground_list = []
        score_list = []
        frame_1 = []
        frame_2 = []

        for i in name_list:
            data_bag = scio.loadmat(self.data_path + 'data' + str(i) + '.mat')
            img = data_bag.get('img'+str(i))
            ground = data_bag.get('ground'+str(i))
            frame1 = data_bag.get('delta1_frame'+str(i))
            frame2 = data_bag.get('delta2_frame'+str(i))

            area_p1 = random.randint(1,80)
            area_p2 = random.randint(1,880)
            if random.randint(0,1) == 1:
                img_list.append(img[area_p1:area_p1+320, area_p2:area_p2+400, :])
                ground_list.append(ground[area_p1:area_p1+320, area_p2:area_p2+400])
                frame_1.append(frame1[area_p1:area_p1+320, area_p2:area_p2+400, :])

                score_list.append(sum(sum(ground[area_p1:area_p1+320, area_p2:area_p2+400])))
                frame_2.append(frame2[area_p1:area_p1+320, area_p2:area_p2+400, :])
            else:
                img_list.append(img[area_p1+319:area_p1-1:-1, area_p2+399:area_p2-1:-1, :])
                ground_list.append(ground[area_p1+319:area_p1-1:-1, area_p2+399:area_p2-1:-1])
                score_list.append(sum(sum(ground[area_p1+319:area_p1-1:-1, area_p2+399:area_p2-1:-1])))
                frame_1.append(frame1[area_p1+319:area_p1-1:-1, area_p2+399:area_p2-1:-1, :])
                frame_2.append(frame2[area_p1+319:area_p1-1:-1, area_p2+399:area_p2-1:-1, :])

            area_p1 = random.randint(320,400)
            area_p2 = random.randint(1,880)
            if random.randint(0,1) == 1:
                img_list.append(img[area_p1:area_p1+320, area_p2:area_p2+400, :])
                ground_list.append(ground[area_p1:area_p1+320, area_p2:area_p2+400])
                frame_1.append(frame1[area_p1:area_p1+320, area_p2:area_p2+400, :])
                score_list.append(sum(sum(ground[area_p1:area_p1+320, area_p2:area_p2+400])))
                frame_2.append(frame2[area_p1:area_p1+320, area_p2:area_p2+400, :])
            else:
                img_list.append(img[area_p1+319:area_p1-1:-1, area_p2+399:area_p2-1:-1, :])
                ground_list.append(ground[area_p1+319:area_p1-1:-1, area_p2+399:area_p2-1:-1])
                frame_1.append(frame1[area_p1+319:area_p1-1:-1, area_p2+399:area_p2-1:-1, :])
                score_list.append(sum(sum(ground[area_p1+319:area_p1-1:-1, area_p2+399:area_p2-1:-1])))
                frame_2.append(frame2[area_p1+319:area_p1-1:-1, area_p2+399:area_p2-1:-1, :])

        score = sorted(enumerate(score_list), key=lambda x:x[1])
        train_batch = np.zeros([self.bsize, self.img_size_x, self.img_size_y,3])
        ground_batch = np.zeros([self.bsize, self.g_size_x, self.g_size_y])
        frame1_batch = np.zeros([self.bsize, self.img_size_x, self.img_size_y, 3])
        frame2_batch = np.zeros([self.bsize, self.img_size_x, self.img_size_y, 3])

        j = 0
        for i in range(len(img_list)):
            if score[i][0]<self.partition[1]:
                train_batch[j,:,:,:] = img_list[i]
                ground_batch[j,:,:] = ground_list[i]
                frame1_batch[j,:,:,:] = frame_1[i]
                frame2_batch[j,:,:,:] = frame_2[i]
                j+=1
                if j==self.bsize:
                    break
            if len(score)-score[i][0]<self.partition[0]:
                train_batch[j,:,:,:] = img_list[i]
                ground_batch[j,:,:] = ground_list[i]
                frame1_batch[j,:,:,:] = frame_1[i]
                frame2_batch[j,:,:,:] = frame_2[i]
                j+=1
                if j==self.bsize:
                    break
        return train_batch,ground_batch,frame1_batch,frame2_batch


