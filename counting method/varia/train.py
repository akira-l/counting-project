import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import config as cfg
from data import Data
from model import net_model
import datetime

class Solver(object):
    
    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.output_dir = os.path.join(cfg.output_dit,datatime.datatime.now().strftime('%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.ckpt_file = os.path.join(self.output_dir,'save.ckpt')
        
        
        
        


def main():
    data = Data()
    train_data, ground_data = data.get_train_data()
    















if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()



