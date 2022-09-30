import os 
import re
import numpy as np
import torch
import pdb
from config import get_args_parser
from torch.utils.data import Dataset
from torch.utils import data
import cv2
from utils import bbox_Oxyxy2normxyxy,bbox_xcychw2xyxy,bbox_xyxy2xcychw,bbox_xyxy2Oxyxy,masks_to_boxes
import torchvision
import math
category2name = {0:'missing_hole',1:'mouse_bite',2:'open_circuit',3:'short',4:'spur',5:'spurious_copper',6:'nonOJ'}
class creat_dataset(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data = data
    def __getitem__(self, index):
        single_data = self.data[index]
        file_name = single_data["file_name"]
        img_path = single_data['img_path']
        bbox = single_data['bbox']
        category = single_data['category']
        img = cv2.resize((cv2.imread(img_path)/255),(600,600))
        img = torch.tensor(img,dtype=torch.float32).to("cuda").permute(2,0,1)
        C,H,W = img.shape
        bbox = torch.tensor(bbox,dtype=torch.long).to("cuda")
        category = torch.tensor(category,dtype=torch.long).to("cuda")
        
        crop_img = torch.zeros([len(bbox),3,224,224],dtype=torch.float32,device="cuda")
        for seq in range(len(bbox)):
            h_2 = (bbox[seq,3]-bbox[seq,1])/2
            w_2 = (bbox[seq,2]-bbox[seq,0])/2
            crop_img[seq,:,math.floor(112-h_2):math.floor(112+h_2),math.floor(112-w_2):math.floor(112+w_2)]= \
                    img[:,bbox[seq,1]:bbox[seq,3],bbox[seq,0]:bbox[seq,2]]
        return crop_img,category
                
    def __len__(self):
        return len(self.data)

class DataLoader(data.DataLoader):
    def __init__(self,src_data,batch_sampler):
        self.src_data = src_data
        self.batch_sampler = batch_sampler
        self.worker_init_fn = 10
        pass
    def __iter__(self):
        
        for batchs in self.batch_sampler:
            imgs = []
            categories = []
            for idx in batchs:
                img,category = self.src_data[idx]
                imgs +=[img]
                categories +=[category]
            imgs = torch.cat(imgs,dim=0)
            categories = torch.cat(categories,dim=0)
            yield imgs,categories
    def __len__(self):
        return len(self.batch_sampler)
    
    
def read_data(args,mode):
    if mode == 'train':
        txt = args.data_foloder+'trainval.txt'
    elif mode == 'val':
        txt = args.data_foloder+'test.txt'
        mode = "test"
    else:
        raise ValueError(f"the mode must be 'train' or 'val', but got unknow value {mode}")    
    with open(txt) as txt_file:
        img_info = []
        for each_txt in txt_file:
            file_name = re.findall('(.*).jpg ',each_txt)[0]
            img_path = args.data_foloder+mode+"/JPEGImages/"+re.findall('(.*.jpg)',each_txt)[0]
            label_path = args.data_foloder+mode+"/Annotations/"+re.findall('jpg (.*)',each_txt)[0]
            
            with open(label_path) as label_file:
                bbox = []
                category = []
                for each_label_txt in label_file:
                    single_label = [int(x) for x in re.findall('(.*) (.*) (.*) (.*) (.*)',each_label_txt)[0]]
                    bbox += [single_label[:-1]]
                    category += [single_label[-1]]
                bbox = np.array(bbox)
                category = np.array(category)
            img_info += [{"file_name":file_name,
                          'img_path':img_path,
                          'bbox':bbox,
                          'category':category
                          }]
    return img_info
    pass
def build_loader(args,mode,data_augmentation=False):
    src_data = read_data(args,mode)
    src_data = creat_dataset(src_data)
    torch.manual_seed(0)
    src_dataS = data.RandomSampler(src_data)
    src_dataS = data.BatchSampler(src_dataS,args.batch_size,drop_last=False)
    data.DataLoader
    src_data = DataLoader(src_data,src_dataS)
    return src_data
    
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    
    train_data = build_loader(args,'train')
    for img,label in train_data:
        # pdb.set_trace()
        test = (img.permute(0,2,3,1).cpu().numpy()*255).astype('uint8').copy()
        for seq in range(len(test)):
            
            print(label[seq])
            # pdb.set_trace()
            cv2.imshow('b',(test[seq]))
            cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('test.png',test)
        
        