import torch
import pdb
import numpy as np
import torchvision
import math
def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    T_S,_,_ = targets.shape
    S_S,_,_ = inputs.shape
    src_mask = inputs[:,None].repeat(1,T_S,1,1).flatten(2,3)
    tgt_mask = targets[None].repeat(S_S,1,1,1).flatten(2,3)
    numerator = (2 * (src_mask * tgt_mask)).sum(-1)
    denominator = src_mask.sum(-1) + tgt_mask.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    
    return loss

def dice_NMS(mask,score,category,IoU_th):

    mask_range = torch.arange(mask.shape[0])

    sorted_score,sorted_score_idx = score.sort(descending=True)

    
    mask = mask[sorted_score_idx]
    score = score[sorted_score_idx]
    category = category[sorted_score_idx]
    
    masked_idx = []
    save_idx = []
    for idx in range(len(mask_range)):
        if idx not in masked_idx:
            # pdb.set_trace()
            diceloss = 1-dice_loss(mask[idx][None],mask).squeeze()
            diceloss[idx] = 0.
            # pdb.set_trace()
            NMS_mask = diceloss>=IoU_th
            masked_idx += list(map(int,mask_range[NMS_mask]))
            save_idx += [idx]
    save_idx = torch.tensor(save_idx)
    
    mask = mask[save_idx]
    score = score[save_idx]
    category = category[save_idx]
    

    return mask,score,category

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

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    # pdb.set_trace()
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    nonOJ_mask = torch.sum(masks,(-2,-1))!=0
    
    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float).to(masks.device)
    x = torch.arange(0, w, dtype=torch.float).to(masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = (x_mask.flatten(1).max(-1)[0]+1).clamp(0,w-1)
    x_min = (x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]-1).clamp(0)

    y_mask = (masks * y.unsqueeze(0))
    y_max = (y_mask.flatten(1).max(-1)[0]+1).clamp(0,h-1)
    y_min = (y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]-1).clamp(0)
    
    bbox = torch.stack([x_min, y_min, x_max, y_max], 1) * nonOJ_mask[:,None]
    return bbox


def xyloc2bbox(proposalXY, src_dxdywh):
    Px,Py = proposalXY.unbind(-1)
    src_dx,src_dy,w,h = src_dxdywh.unbind(-1)
    
    new_x = src_dx*w + Px
    new_y = src_dy*h + Py
    
    bbox = torch.stack([new_x,new_y,w,h],dim=-1)
    return bbox
def loc2bbox(src_xcychw, loc):

    src_xc, src_yc, src_w, src_h = src_xcychw.unbind(-1)
    
    dx,dy,dw,dh = loc.unbind(-1)

    new_xc = dx * src_w + src_xc
    new_yc = dy * src_h + src_yc
    new_w = torch.exp(dw) * src_w
    new_h = torch.exp(dh) * src_h
    dst_bbox = torch.stack([new_xc, new_yc, new_w, new_h],dim=-1)

    return dst_bbox
    

def bbox2loc(src_xcychw, tgt_xcychw):
    src_xc, src_yc, src_w, src_h = src_xcychw.unbind(-1)
    tgt_xc, tgt_yc, tgt_w, tgt_h = tgt_xcychw.unbind(-1)
    
    dx = (tgt_xc-src_xc)/src_w
    dy = (tgt_yc-src_yc)/src_h
    
    dw = torch.log(tgt_w/src_w)
    dh = torch.log(tgt_h/src_w)
    
    return torch.stack([dx,dy,dw,dh],dim=-1)
    
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

def CIoU(src_xcycwh, tgt_xcycwh):
    eps = torch.finfo(src_xcycwh.dtype).eps
    src_xc, src_yc, src_w, src_h = src_xcycwh.unbind(-1)
    tgt_xc, tgt_yc, tgt_w, tgt_h = tgt_xcycwh.unbind(-1)
    
    src_bbox = bbox_xcychw2xyxy(src_xcycwh)
    tgt_bbox = bbox_xcychw2xyxy(tgt_xcycwh)
    
    p_2 = (src_xc-tgt_xc)**2+(src_yc-tgt_yc)**2
    
    lt = torch.min(src_bbox[:,:2],tgt_bbox[:,:2])
    rb = torch.max(src_bbox[:,2:],tgt_bbox[:,2:])
    
    c_2 = torch.sum((lt-rb)**2,dim=-1)
    
    v = (4/math.pi) * (torch.arctan(tgt_w/tgt_h) - torch.arctan(src_w/src_h))**2
    
    iou, _ = box_iou(src_bbox,tgt_bbox)
    iou = torch.diag(iou)
    a = v/((1-iou)+v).clamp(eps)
    
    CIoUloss = 1 - iou + p_2/c_2 + a*v
    
    return CIoUloss

def bbox_area(boxes):
    sx, sy, lx, ly = boxes.unbind(-1)
    xlong = lx-sx
    ylong = ly-sy
    return xlong*ylong


def load_model_parameter(file_path,model,optimizer,load_dict,device,mode_name):

    dict_train_record = {'start_epoch':None,'end_epoch':None,
                         'train_losses':None,'val_losses':None,
                         'lr':None,
                         'train_acc':None,'val_acc':None}
    try:
        dict_ckpt = torch.load(file_path,map_location=device)
    except:
        raise ValueError("the ckpt path or model has some wrong, please check it")
    if mode_name=="segmentation":
        model[0].load_state_dict(dict_ckpt["backbone"+'model'])
        model[1].load_state_dict(dict_ckpt[mode_name+'model'])
    else:
        # pdb.set_trace()
        model.load_state_dict(dict_ckpt[mode_name+'model'])
        
        # torch.save({"backbonemodel":dict_ckpt["model"],
        #             'optimizer': dict_ckpt["optimizer"],
        #             'start_epoch': dict_ckpt["start_epoch"],
        #             'lr': dict_ckpt["lr"],
        #             'end_epoch': dict_ckpt["end_epoch"],
        #             'train_losses': dict_ckpt["train_losses"],
        #             'val_losses': dict_ckpt["val_losses"],
        #             'train_acc': dict_ckpt["train_acc"],
        #             'val_acc': dict_ckpt["val_acc"],
        #             # "train_conf_acc": dict_ckpt["train_conf_acc"],
        #             # "val_conf_acc": dict_ckpt["val_conf_acc"]},
        #             args.save_model_path+"aaa")
        
                        
        # dict_keys(['model', 'optimizer', 'start_epoch', 'lr', 'end_epoch',
        #            'train_losses', 'val_losses', 'train_acc', 'val_acc'])
    if optimizer != False:
        optimizer.load_state_dict(dict_ckpt['optimizer'])
    if load_dict:
        for n,v in dict_train_record.items():
                dict_train_record[n] = dict_ckpt[n]
    print("="*100)
    print(f"{mode_name} loaded weight: "+file_path)
    print("="*100)
    return model,optimizer,dict_train_record



if __name__ == "__main__":
    tensor = torch.zeros([1,1,10,10])
    tensor[:,:,5:9,4:7] = 1.
    print(tensor)
    print(masks_to_boxes(tensor, 0.4))
    pass