import torch
import pdb
import time
import re
from utils import smooth_value
class Logger():
    def __init__(self,print_freq,dataset,batch_size,not_show_list,not_smooth_list,epochs):
        super().__init__()
        self.print_freq = print_freq
        self.update_count = 0
        self.data_len = len(dataset)
        self.batch_size = batch_size
        self.epochs = epochs
        self.record = {}
        self.starttime = time.time()
        self.endtime = 0
        self.runtime = 0
        self.eta = 0
        self.not_show_list = not_show_list
        self.not_smooth_list = not_smooth_list
    def update(self,record,epoch,mode):
        self.mode = mode
        self.update_count +=1
        self.runtime = (time.time()-self.starttime)/self.update_count
        self.endtime = self.runtime*self.data_len
        for key in record.keys():
            if key in self.not_show_list:
                continue
            if key not in self.record.keys():
                self.record[key] = 0.
            
            self.record[key] = list(map(float,record[key]))

        self.epoch = epoch
        if (self.update_count%self.print_freq) == 0 or self.update_count==self.data_len:
            print_str = ''
            for key,value in self.record.items():
                if key not in self.not_smooth_list:
                    print_str += f'[{key}: {round(value[-1],3)}({round(smooth_value(value),3)})]'

                else:
                    if 'lr' in key:
                        lr = ''
                        if re.findall('(.*\.\d{3}).*(e.*)',str(value[-1])) != []:
                            for lr_str in re.findall('(.*\.\d{3}).*(e.*)',str(value[-1]))[0]:
                                lr += lr_str
                        else:
                            lr = str(value)[0:7]
                        print_str += f'[{str(key)}: {lr}]'

            
            self.__print_Logger(print_str)
        pass
    def __print_Logger(self,print_str):
        print('\r'+' '*len(print_str),end='')
        train_now_data = self.update_count
        progress_bar = ''
        self.eta = 'eta:{'+str(round(self.runtime*self.update_count,1))+f'/{round(self.endtime,1)}'+'}'
        if train_now_data != self.data_len:
            progress_bar += '█'*int((train_now_data/self.data_len)*10)
            progress_bar +=' '*(10-len(progress_bar))
            end = ''
        else:
            progress_bar = '█'*10
            end = '\n'
            # self.__reset_log()
        print(f'\r{self.mode} epoch:{self.epoch}/{self.epochs}|{progress_bar}|[{train_now_data}/{self.data_len}]---'+self.eta+print_str,end=end)
        pass

    def reset_log(self):
        self.record = {}
        self.update_count = 0
        self.starttime = time.time()
        self.endtime = 0
        self.runtime = 0
        
if __name__ =='__main__':
    loss_log = Logger(1,range(1000),10,range(1000))
    record = {'1':0,'2':0}
    for i in range(1000):
        for j in range(1000):
            time.sleep(0.001)
            record['1'] +=torch.tensor(i)
            record['2'] +=torch.tensor(i+1)
            loss_log.update(record,i)
