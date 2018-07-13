    def get_train_data(self):

        need_img = sum(self.partition)
        need_img = (need_img//self.data_get_from_one_img)+1
        for i in range(need_img):
            data_num = self.seq[self.batch_times]
            data_baggage = scio.loadmat(self.data_path + 'data' + str(data_num) + '.mat')
            img = data_baggage.get('img',str(data_num))
            ground = data_baggage.get('ground'+str(data_num))
            self.img_size_x = 400
            self.img_size_y = 320
            self.g_size_x = 
            self.g_size_y = 
            self.data_cap_map = [[0,400,400,720],[880,1280,400,720],[71,471,21,341],[596,996,31,351],[487,887,360,680]]
            img_list = []
            ground_list = []
            score_list = []
            #x 30 pixel random with last 3 areas
            for j in range(need_img):
                for k in range(len(self.data_cap_map)):
                    area = self.data_cap_map[k]
                    img_list.append(img_list[j[[area[0]:area[1],area[2]:area[3],:])
                    ground_list.append(ground_list[j][area[0]:area[1],area[2]:area[3]])
                    score_tmp.append(sum(sum(ground_list[j][area[0]:area[1],area[2]:area[3]])))
                    
                score = sorted(enumerate(score_tmp), key=lambda x:x[1])
                
                
            
            
        
        
        
