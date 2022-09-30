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
category2name = {0:'missing_hole',1:'mouse_bite',2:'open_circuit',3:'short',4:'spur',5:'spurious_copper',6:'nonOJ'}
class creat_dataset(Dataset):
    def __init__(self,data,data_augmentation):
        super().__init__()
        self.data_augmentation = data_augmentation
        self.data = data
        self.RHF = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.RVF = torchvision.transforms.RandomVerticalFlip(p=0.5)
        self.RRT = torchvision.transforms.RandomRotation(180, center=(320, 320))
        self.RC = torchvision.transforms.RandomCrop([512,512], fill=0, padding_mode='constant')
    def __getitem__(self, index):
        single_data = self.data[index]
        file_name = single_data["file_name"]
        img_path = single_data['img_path']
        bbox = single_data['bbox']
        category = single_data['category']
        img = cv2.resize((cv2.imread(img_path)/255),(600,600))
        img = torch.tensor(img,dtype=torch.float32).to("cuda").permute(2,0,1)
        C,H,W = img.shape
        Original_mask = torch.zeros([len(bbox),600,600],device="cuda")
        mask = Original_mask
        for seq in range(len(bbox)):
            Original_mask[seq,bbox[seq,1]:bbox[seq,3],
                          bbox[seq,0]:bbox[seq,2]] = 1.
            
        if self.data_augmentation:
            cat_img = torch.cat([img,Original_mask],dim=0)
            cat_img = self.RHF(cat_img)
            crop_img = self.RVF(cat_img)
            crop_img = self.RRT(crop_img)
            img = crop_img[0:3]
            mask = crop_img[3:]
            if mask.dim()==2:
                mask = mask.unsqueeze(0)
        area_mask = torch.sum(mask,dim=(-2,-1))
        area_Omask = torch.sum(Original_mask,dim=(-2,-1))
        # pdb.set_trace()
        intersetion_th = (area_mask/area_Omask)>=0.5
        # pdb.set_trace()
        mask = mask[intersetion_th]
        bbox = bbox_xyxy2xcychw(bbox_Oxyxy2normxyxy(masks_to_boxes(mask),(W,H)))
        category = torch.tensor(category,dtype=torch.long).to("cuda")[intersetion_th]
        # pdb.set_trace()
        
        return img,{"file_name":file_name,'bbox':bbox,'category':category}
                
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
            labels = []
            for idx in batchs:
                img,label = self.src_data[idx]
                imgs +=[img]
                labels +=[label]
            imgs = torch.stack(imgs,dim=0)

            yield imgs,labels
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
    src_data = creat_dataset(src_data,data_augmentation)
    torch.manual_seed(0)
    src_dataS = data.RandomSampler(src_data)
    src_dataS = data.BatchSampler(src_dataS,args.batch_size,drop_last=False)
    data.DataLoader
    src_data = DataLoader(src_data,src_dataS)
    return src_data
    
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    
    train_data = build_loader(args,'train',True)
    for img,label in train_data:
        # pdb.set_trace()
        test = (img.permute(0,2,3,1).cpu().numpy()*255).astype('uint8')[0].copy()
        H,W,C = test.shape
        bbox = bbox_xyxy2Oxyxy(bbox_xcychw2xyxy(label[0]['bbox']),(H,W)).long().cpu().numpy()
        S,_ = bbox.shape
        
        category = label[0]['category']
        # print(category)
        for seq in range(S):
            cv2.rectangle(test,[bbox[seq,0],bbox[seq,1]],[bbox[seq,2],bbox[seq,3]],[0,0,255],3)
            cv2.rectangle(test,[bbox[seq,0],bbox[seq,1]-25],[bbox[seq,0]+len(category2name[category[seq].item()])*12,
                                                              bbox[seq,1]-5],[255,255,255],-1)
            cv2.rectangle(test,[bbox[seq,0],bbox[seq,1]-25],[bbox[seq,0]+len(category2name[category[seq].item()])*12,
                                                              bbox[seq,1]-5],[0,0,0],1)
            cv2.putText(test, category2name[category[seq].item()], [bbox[seq,0],bbox[seq,1]-10], cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (150, 0, 255), 1, cv2.LINE_AA)
            
        # for seq in range(S):
        #     test[:,:,2] += mask[:,:,seq]
        cv2.imshow('b',(test))
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('test.png',test)
        
        