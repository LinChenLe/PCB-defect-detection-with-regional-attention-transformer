from config import get_args_parser
from dataloader import build_loader,DataLoader
import torch
from torch import nn
import cv2
from torch.utils import data
from creat_model import build_segmentation_model
import numpy as np
import pdb
from setcriterion import Hungarian_match
import copy
import torchvision
from utils import masks_to_boxes,bbox_Oxyxy2normxyxy,bbox_xcychw2xyxy,\
    bbox_xyxy2xcychw,bbox_xyxy2Oxyxy,bbox_area,NMS,load_model_parameter,dice_NMS,xyloc2bbox

torch.manual_seed(0)
category2name = {0:'open',1:'short',2:'bite',3:'supr',4:'copper',5:'hole',6:'nonOJ'}
# conf_th = 0.1
class_th = 0.0
# mask_th = 0.4
# label_filter = 5
@torch.no_grad()
def inference(dataloader: data.Dataset, segmentation_model: nn.Module,
              device: torch.device, with_tgt: bool=False, threshold:int=0.6):
    segmentation_model.eval()
    softmax = nn.Softmax(dim=-1)
    for idx,(template,tested,tgt) in enumerate(dataloader):
        N,C,H,W = template.shape
        template = template.to(device)
        tested = tested.to(device)
        for batch in range(N):
            tgt[batch]['mask'] = tgt[batch]['mask'].to(device)
            tgt[batch]['category'] = tgt[batch]['category'].to(device)
        
        segmen_result = segmentation_model(template,tested)
        segmen_PC = torch.stack([x["proposal_center"] for x in segmen_result],dim=0)
        segmen_bbox = torch.stack([x["bbox"] for x in segmen_result],dim=0)
        segmen_category = softmax(torch.stack([x["category"] for x in segmen_result],dim=0))
        # segmen_conf = torch.stack([x["conf"] for x in segmen_result],dim=0).squeeze()

        
        for batch in range(N):
            file_name = tgt[batch]["file_name"]
            print("processing:",file_name)
            # batch_conf = segmen_conf[batch][:,-1]
            bboxes = bbox_xyxy2Oxyxy(bbox_xcychw2xyxy(xyloc2bbox(segmen_PC[batch],segmen_bbox[batch])), (W,H))
            classify_result = segmen_category[batch]
            # pdb.set_trace()
            maxscore,maxidx = torch.max(classify_result,dim=-1)
            # pdb.set_trace()
            classify_mask = (maxscore>=class_th) * (maxidx != 6)# * (batch_conf>conf_th)
            if torch.sum(classify_mask) == 0:
                continue
            maxidx = maxidx[classify_mask]
            maxscore = maxscore[classify_mask]
            bboxes = bboxes[classify_mask]
            # batch_conf = batch_conf[classify_mask]
            bboxes,maxscore,maxidx = NMS(bboxes,maxscore,maxidx,0.3)
            bboxes = bboxes.to(torch.long)
            maxidx = maxidx.cpu().numpy()
            
            
            
            with open(f"Answer/DeepPCB/submit/{file_name}.txt","w+") as file:
                for seq in range(len(bboxes)):
                    box = bboxes[seq]
                    mi = maxidx[seq].item()+1
                    ms = maxscore[seq].item()
                    putStr = ""
                    for coord in box:
                        putStr+= str(coord.item())+","
                    putStr += str(round(ms,2))+','+str(mi)+"\n"
                    file.write(putStr)
            
            
            
        
    
        

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    backbone_model,segmentation_model = build_segmentation_model(args)
    if args.load_ckpt != '' and args.load_ckpt != 'ckpt/':
        model,segmentation_optimizer,record = load_model_parameter(args.load_ckpt,
                                            [backbone_model,segmentation_model],False,True,args.device,
                                            mode_name="segmentation")
        backbone_model = model[0]
        segmentation_model = model[1]
        

    train_data = build_loader(args,'val')

    

    
    inference(train_data,segmentation_model,device,with_tgt=True)
    
    pass