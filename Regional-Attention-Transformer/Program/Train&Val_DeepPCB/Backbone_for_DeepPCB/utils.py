import torch
import pdb
import numpy as np
import torchvision
def NMS(bbox,score,category,IoU_th):
    try:
        sele = torchvision.ops.nms(bbox, score,IoU_th)
    except:
        pdb.set_trace()
    bbox = bbox[sele]
    score = score[sele]
    category = category[sele]
    return bbox,score,category
    pass
# def masks_to_boxes(mask,th):
#     relu = torch.nn.ReLU()
#     if len(mask.shape) == 4:
#         resultN,resultC,resultH,resultW = mask.shape
#     elif len(mask.shape) == 3:
#         resultC,resultH,resultW = mask.shape
#     box_mask = relu(mask-th)
#     xscore,_ = torch.max(box_mask,dim=-2)
#     yscore,_ = torch.max(box_mask,dim=-1)
    
#     xlfidx = torch.max(xscore>0,dim=-1)[1]
#     width = torch.sum(xscore>0,dim=-1)
    
#     ylfidx = torch.max(yscore>0,dim=-1)[1]
#     hight = torch.sum(yscore>0,dim=-1)
    
#     xlfidx = xlfidx-1
#     xrbidx = xlfidx+width+1
    
#     ylfidx = ylfidx-1
#     yrbidx = ylfidx+hight+1
    
#     bbox = bbox_xyxy2xcychw(torch.stack([xlfidx,ylfidx,xrbidx,yrbidx],dim=-1).clamp(0,resultH)/resultH)
#     # bbox = torch.stack([xlfidx,ylfidx,xrbidx,yrbidx],dim=-1).clamp(0,resultH)
#     return bbox

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    nonOJ_mask = torch.sum(masks,(-2,-1))!=0
    
    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float).to(masks.device)
    x = torch.arange(0, w, dtype=torch.float).to(masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
    
    bbox = torch.stack([x_min, y_min, x_max, y_max], 1) * nonOJ_mask[:,None]
    return bbox


    
def smooth_value(value:list):
    SV = sum(value)/len(value)
    return SV
def bbox_xyxy2xcychw(bboxes):
    sx,sy,ex,ey = bboxes.unbind(dim=-1)
    w = ex - sx
    h = ey - sy
    
    return torch.stack([sx+0.5*w,sy+0.5*h,
                        w,       h],dim=-1)
def bbox_xcychw2xyxy(bboxes):
    xc,yc,w,h = bboxes.unbind(dim=-1)

    
    return torch.stack([xc-0.5*w,yc-0.5*h,
                        xc+0.5*w,yc+0.5*h],dim=-1)

def bbox_xyxy2Oxyxy(bboxes,WH):
    sx,sy,ex,ey = bboxes.unbind(dim=-1)
    
    sx = sx*WH[0] 
    ex = ex*WH[0]
    sy = sy*WH[1]
    ey = ey*WH[1]
    
    return torch.stack([sx,sy,
                        ex,ey],dim=-1)

def bbox_Oxyxy2normxyxy(bboxes,WH):

    sx,sy,ex,ey = bboxes.unbind(dim=-1)
    sx = sx/WH[0]
    sy = sy/WH[1]
    ex = ex/WH[0]
    ey = ey/WH[1]

    return torch.stack([sx,sy,
                        ex,ey],dim=-1)

def box_iou(src_xyxy, tgt_xyxy):
    src_area = bbox_area(src_xyxy)
    tgt_area = bbox_area(tgt_xyxy)

    lt = torch.max(src_xyxy[:,None,:2], tgt_xyxy[:,:2])
    rb = torch.min(src_xyxy[:,None,2:], tgt_xyxy[:,2:])

    wh = (rb - lt).clamp(min=0)

    inter = wh[:, :, 0] * wh[:, :, 1]

    union = src_area[:, None] + tgt_area - inter

    iou = inter / union
    return iou, union

def box_iou_npV(src_xyxy, tgt_xyxy):
    src_xyxy = np.array(src_xyxy)
    tgt_xyxy = np.array(tgt_xyxy)
    s_lx = src_xyxy[2]-src_xyxy[0]
    s_ly = src_xyxy[3]-src_xyxy[1]
    
    t_lx = tgt_xyxy[2]-tgt_xyxy[0]
    t_ly = tgt_xyxy[3]-tgt_xyxy[1]
    
    src_area = s_lx*s_ly
    tgt_area = t_lx*t_ly 
    
    lt = np.max([src_xyxy[:2],tgt_xyxy[:2]],axis=0)
    rb = np.min([src_xyxy[2:], tgt_xyxy[2:]],axis=0)

    wh = (rb - lt).clip(min=0)
    
    inter = wh[0] * wh[1]

    union = src_area + tgt_area - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(src_xyxy, tgt_xyxy):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """

    assert (src_xyxy[:, 2:] >= src_xyxy[:, :2]).all()
    assert (tgt_xyxy[:, 2:] >= tgt_xyxy[:, :2]).all()

    iou, union = box_iou(src_xyxy, tgt_xyxy)

    lt = torch.min(src_xyxy[:, None, :2], tgt_xyxy[:, :2])
    rb = torch.max(src_xyxy[:, None, 2:], tgt_xyxy[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]

    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
def bbox_area(boxes):
    sx, sy, lx, ly = boxes.unbind(-1)
    xlong = lx-sx
    ylong = ly-sy
    return xlong*ylong

def bbox_area(boxes):
    sx, sy, lx, ly = boxes.unbind(-1)
    xlong = lx-sx
    ylong = ly-sy
    return xlong*ylong

def load_model_parameter(file_path,model,optimizer,device):

    dict_train_record = {'start_epoch':None,'end_epoch':None,
                         'train_losses':None,'val_losses':None,
                         'lr':None,
                         'train_acc':None,'val_acc':None}
    dict_ckpt = torch.load(file_path,map_location=device)
    # pdb.set_trace()
    model.load_state_dict(dict_ckpt['backbonemodel'])
    optimizer.load_state_dict(dict_ckpt['optimizer'])

    for n,v in dict_train_record.items():
            dict_train_record[n] = dict_ckpt[n]

    return model,optimizer,dict_train_record
    pass

if __name__ == "__main__":
    tensor = torch.zeros([1,1,10,10])
    tensor[:,:,5:9,4:7] = 1.
    print(tensor)
    print(masks_to_boxes(tensor, 0.4))
    pass