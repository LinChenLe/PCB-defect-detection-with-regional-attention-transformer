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
    bbox_xyxy2xcychw,bbox_xyxy2Oxyxy,bbox_area,NMS,load_model_parameter,dice_NMS,xyloc2bbox,box_iou
torch.manual_seed(0)
category2name = {-1:"nonDetect",0:'open',1:'short',2:'bite',3:'supr',4:'copper',5:'hole',6:'nonOJ'}
Error2Color = {"both":{"bboxC":[0,0,255],"categoryC":[0,0,255]},
               "category":{"bboxC":[0,255,0],"categoryC":[0,0,255]},
               "bbox":{"bboxC":[0,0,255],"categoryC":[0,255,0]},
               "Correct":{"bboxC":[0,255,0],"categoryC":[0,255,0]},
               "NoneDetect":{"bboxC":[240,32,160],"categoryC":[240,32,160]}}
Error_count = {"category":0,
               "bbox":0,
               "Correct":0,
               "NoneDetect":0,
               "total_Defect":0}
area_th = 0
class_th = 0.0
iou_th = 0.33
@torch.no_grad()
def inference(dataloader: data.Dataset, segmentation_model: nn.Module,
              device: torch.device):
    segmentation_model.eval()
    softmax = nn.Softmax(dim=-1)
    for idx,(template,tested,tgt) in enumerate(dataloader):
        N,C,H,W = template.shape
        template = template.to(device)
        tested = tested.to(device)
        for batch in range(N):
            tgt[batch]['bbox'] = tgt[batch]['bbox'].to(device)
            tgt[batch]['category'] = tgt[batch]['category'].to(device)
            Error_count["total_Defect"] += len(tgt[batch]['category'])
        segmen_result = segmentation_model(template,tested)
        segmen_PC = torch.stack([x["proposal_center"] for x in segmen_result],dim=0)
        segmen_bbox = torch.stack([x["bbox"] for x in segmen_result],dim=0)
        segmen_category = softmax(torch.stack([x["category"] for x in segmen_result],dim=0))
        
        for batch in range(N):
            file_name = tgt[batch]["file_name"]
            print("processing:",file_name)
            src_bboxes = bbox_xyxy2Oxyxy(bbox_xcychw2xyxy(xyloc2bbox(segmen_PC[batch],segmen_bbox[batch])), (W,H))
            tgt_bboxes = bbox_xyxy2Oxyxy(bbox_xcychw2xyxy(tgt[batch]["bbox"]),(W,H))
            src_category = segmen_category[batch]
            tgt_category = tgt[batch]["category"]
            src_conf,src_category = torch.max(src_category,dim=-1)
            
            classify_mask = (src_conf>=class_th) * (src_category != 6)
           
            src_category = src_category[classify_mask]
            src_conf = src_conf[classify_mask]
            src_bboxes = src_bboxes[classify_mask]
            src_bboxes,src_conf,src_category = NMS(src_bboxes,src_conf,src_category,0.5)
            src_bboxes = src_bboxes.to(torch.long)

            
            src_conf,src_conf_idx = src_conf.sort(descending=True)
            
            iou = box_iou(src_bboxes,tgt_bboxes)[0]
            
            src_bboxes = src_bboxes[src_conf_idx]
            src_category = src_category[src_conf_idx]
            match_iou = iou[src_conf_idx]
            match_mask = match_iou>iou_th
            print_img = False
            img = tested[batch].repeat(3,1,1).permute(1,2,0).cpu().numpy()
            img = (img.copy()*255).astype('uint8')
            for idx in range(match_mask.shape[0]):
                match_gt_idx = torch.argmax(match_iou[idx])
                
                category_mask = tgt_category[match_gt_idx] == src_category[idx]
                detected = match_mask[idx,match_gt_idx].clone()
                if detected != False:
                    match_mask[:,match_gt_idx] = False
                    match_mask[idx,match_gt_idx] = detected
                
                ToF = category_mask*match_mask[idx,match_gt_idx]
                

                show_bbox = src_bboxes[idx].to(torch.long).cpu().numpy()
                show_conf = src_conf[idx].to(torch.float32).cpu().numpy()
                show_category = src_category[idx].to(torch.long).cpu().numpy()

                if category_mask == False and match_mask[idx,match_gt_idx] == False:
                    Error = "both"
                    Error_count["bbox"] += 1
                    Error_count["category"] += 1
                elif category_mask == False:
                    Error = "category"
                    Error_count["category"] += 1
                elif match_mask[idx,match_gt_idx] == False:
                    Error = "bbox"
                    Error_count["bbox"] += 1
                else:
                    Error = "Correct"
                    Error_count["Correct"] += 1
                drow_img(img,show_bbox,show_conf,show_category,Error)
                
                if ToF == False:
                    print_img = True
            
            
            tgt_mask = torch.sum(match_mask,0) != 1
            if torch.sum(tgt_mask)>0:
                if ToF == False:
                    print_img = True
                Error = "NoneDetect"
                
                for idx in range(tgt_mask.shape[-1]):
                    
                    if tgt_mask[idx]==True:
                        Error_count["NoneDetect"] += 1
                        show_bbox = tgt_bboxes[tgt_mask][0].to(torch.long).cpu().numpy()
                        show_category = -1
                        show_conf = 0.00
                        drow_img(img,show_bbox,show_conf,show_category,Error)
  

            if print_img:
                cv2.imwrite(f"error/{file_name}.jpg",img)
    with open("error/log.txt","w+") as log_file:
        putStr = ""
        for key in Error_count.keys():
            putStr += f"{key}:{Error_count[key]}\n"
        log_file.write(putStr)


def drow_img(img,bbox,conf,category,Error):
    Color = Error2Color[Error]
    category_str = category2name[int(category)]
    cv2.rectangle(img,[bbox[0],bbox[1]],[bbox[2],bbox[3]],Color["bboxC"],2)
    cv2.rectangle(img,[bbox[0],bbox[1]-20],[bbox[0]+len(category_str+f":{conf:.2f}")*10,bbox[1]-4],[255,255,255],-1)
    cv2.rectangle(img,[bbox[0],bbox[1]-20],[bbox[0]+len(category_str+f":{conf:.2f}")*10,bbox[1]-4],Color["categoryC"],2)

    cv2.putText(img, category_str+f":{conf:.2f}", [bbox[0],bbox[1]-5], cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (150, 0, 255), 1, cv2.LINE_AA)
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    backbone_model,segmentation_model = build_segmentation_model(args)
    if args.load_ckpt != '' and args.load_ckpt != 'ckpt/':
        model,_,_ = load_model_parameter(args.load_ckpt,
                                         [backbone_model,segmentation_model],False,True,args.device,
                                         mode_name="segmentation")
        backbone_model = model[0]
        segmentation_model = model[1]
        

    train_data = build_loader(args,'val')

    

    
    inference(train_data,segmentation_model,device)
    
    pass