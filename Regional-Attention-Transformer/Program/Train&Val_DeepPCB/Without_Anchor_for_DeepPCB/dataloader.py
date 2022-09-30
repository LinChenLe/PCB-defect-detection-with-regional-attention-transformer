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
category2name = {0:'open',1:'short',2:'bite',3:'supr',4:'copper',5:'hole',6:'nonOJ'}
class creat_dataset(Dataset):
    def __init__(self,data,model_out_size,data_augmentation):
        super().__init__()
        self.data = data
        self.model_out_size = model_out_size
        self.data_augmentation = data_augmentation
        self.RHF = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.RVF = torchvision.transforms.RandomVerticalFlip(p=0.5)
        self.RRT = torchvision.transforms.RandomRotation(180, center=(320, 320))
        self.RC = torchvision.transforms.RandomCrop([512,512], fill=0, padding_mode='constant')
        self.idx = 0
    def __getitem__(self, index):
        single_data = self.data[index]
        file_name = single_data["file_name"]
        temp_path = single_data['temp_path']
        test_path = single_data['test_path']
        mask_path = single_data['mask_path']
        mask_category_path = single_data['mask_category_path']
        temp_img = (cv2.imread(temp_path,0)/255)[np.newaxis,:,:]
        temp_img = torch.tensor(temp_img,dtype=torch.float32).to("cuda")
        
        test_img = (cv2.imread(test_path,0)/255)[np.newaxis,:,:]
        test_img = torch.tensor(test_img,dtype=torch.float32).to("cuda")
        
        Original_mask = torch.tensor((np.load(mask_path)/255).transpose(2,0,1),dtype=torch.float32).to("cuda")
        mask = Original_mask
        mask_category = (torch.tensor(np.load(mask_category_path),dtype=torch.long)-1).to("cuda")
        _,H,W = Original_mask.shape
        if self.data_augmentation:
            cat_img = torch.cat([temp_img,test_img,Original_mask],dim=0)
            cat_img = self.RHF(cat_img)
            crop_img = self.RVF(cat_img)
            crop_img = self.RRT(crop_img)
            # if (self.idx%4)==0:
            #     random_num = torch.randint(low=480,
            #                                 high=640,
            #                                 size=[2])
            #     self.RC = torchvision.transforms.RandomCrop([random_num[0],random_num[1]])
                
            #     self.idx = 0
            # crop_img = self.RC(crop_img)
            # pdb.set_trace()

# =============================================================================

            # crop_img = self.RC(cat_img)
# =============================================================================
            temp_img = crop_img[0][None]
            test_img = crop_img[1][None]
            mask = crop_img[2:]
            non_mask = torch.sum(mask,dim=(-2,-1))!=0
            non_mask[-1] = True
            # mask = mask[non_mask]
            # mask_category = mask_category[non_mask[:-1]]
            # Ms,H,W = mask.shape
            # if Ms == 1:
            #     zeros = torch.zeros([1,H,W],dtype=torch.float32)
            #     mask = torch.cat([zeros,mask],dim=0)

        area_mask = torch.sum(mask,dim=(-2,-1))
        area_Omask = torch.sum(Original_mask,dim=(-2,-1))
        # pdb.set_trace()
        intersetion_th = (area_mask/area_Omask)>=0.5
        intersetion_th[-1] = True
        
        mask = mask[intersetion_th]
        mask_category = mask_category[intersetion_th[:-1]]
        bbox = bbox_xyxy2xcychw(bbox_Oxyxy2normxyxy(masks_to_boxes(mask[:-1]),(W,H)))
        mask = torchvision.transforms.Resize((int(H/self.model_out_size),int(W//self.model_out_size)))(mask)
        mask[mask!=0] = 1.

        return temp_img,test_img,{"file_name":file_name,'mask':mask,'bbox':bbox,'category':mask_category}
                
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
            templates = []
            testeds = []
            labels = []
            for idx in batchs:
                template,tested,label = self.src_data[idx]
                templates +=[template]
                testeds +=[tested]
                labels +=[label]
            templates = torch.stack(templates,dim=0)
            testeds = torch.stack(testeds,dim=0)

            yield templates,testeds,labels
    def __len__(self):
        return len(self.batch_sampler)
    
    
def read_data(args,mode):
    if mode == 'train':
        txt = args.data_foloder+'trainval.txt'
    elif mode == 'val':
        txt = args.data_foloder+'test.txt'
    else:
        raise ValueError(f"the mode must be 'train' or 'val', but got unknow value {mode}")    
    with open(txt) as txt_file:
        img_info = []
        for each_txt in txt_file:
            file_name = re.findall('(\d{8}.*).jpg',each_txt)[0]
            img_path = args.data_foloder+re.findall('(.*).jpg',each_txt)[0]
            label_path = args.data_foloder+re.findall('jpg (.*)',each_txt)[0]
            mask_path = img_path+'_mask.npy'
            mask_category_path = img_path+'_category.npy'
            temp_path = img_path+'_temp.jpg'
            test_path = img_path+'_test.jpg'
            
            with open(label_path) as label_file:
                bbox = []
                category = []
                for each_label_txt in label_file:
                    single_label = [int(x) for x in re.findall('(.*) (.*) (.*) (.*) (.*)',each_label_txt)[0]]
                    bbox += [single_label[:-1]]
                    category += [single_label[-1]-1]
                bbox = np.array(bbox)
                category = np.array(category)
            img_info += [{"file_name":file_name,
                          'temp_path':temp_path,
                          'test_path':test_path,
                          'mask_path':mask_path,
                          'mask_category_path':mask_category_path,
                          'bbox':bbox,
                          'category':category
                          }]
    return img_info
    pass
def build_loader(args,mode,data_augmentation=False):
    src_data = read_data(args,mode)
    src_data = creat_dataset(src_data,args.model_out_size,data_augmentation)
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
    for temp,test,label in train_data:

        test = (test.permute(0,2,3,1).repeat(1,1,1,3).numpy()*255).astype('uint8')[0].copy()
        H,W,C = test.shape

        mask = cv2.resize((label[0]['mask'].permute(1,2,0).numpy()*255).astype('uint8').copy(),(H,W))
        bbox = bbox_xyxy2Oxyxy(bbox_xcychw2xyxy(label[0]['bbox']),(H,W)).long().numpy()
        S,_ = bbox.shape
        category = label[0]['category']

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
        
        