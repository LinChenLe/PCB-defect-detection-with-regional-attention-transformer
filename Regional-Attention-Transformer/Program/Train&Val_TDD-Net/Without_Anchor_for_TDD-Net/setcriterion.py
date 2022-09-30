from torch import nn
import torch
from torch.nn.functional import binary_cross_entropy
from utils import bbox_Oxyxy2normxyxy,bbox_xcychw2xyxy,bbox_xyxy2xcychw,\
    bbox_xyxy2Oxyxy,bbox_area,masks_to_boxes,box_iou,bbox2loc,generalized_box_iou,xyloc2bbox,CIoU
from scipy.optimize import linear_sum_assignment
import pdb
import numpy as np
import torch.nn.functional as F
from torch.utils import data
import math
class Hungarian_match(nn.Module):
    def __init__(self,device,iou_th=0.4):
        super().__init__()
        self.device=device
        self.softmax = nn.Softmax(dim=-1)
        self.MSE = nn.MSELoss(reduction='none')
        self.iou_th = iou_th
        self.a = 0.8
    @torch.no_grad()
    # Hungarian(segmen_result,mask)
    def forward(self,src,tgt):
        '''
        src = {'mask': for each defect, 'category': [probability category]}
        tgt = ['mask': for each defect, 'category': [category]}
        '''
        
        N = len(src)
        # src_mask = [x['mask'] for x in src]
        # tgt_mask = [x['mask'][:-1] for x in tgt]
        src_PC = [x["proposal_center"] for x in src]
        src_bbox = [x["bbox"] for x in src]
        tgt_bbox = [x["bbox"] for x in tgt]

        tgt_category = [x['category'] for x in tgt]
        # bbox = masks_to_boxes(torch.stack(src_mask),0.4)
        assert(len(src_bbox)==len(tgt_bbox)),f"the src batch size[{len(src_bbox)}] not equal to tgt batch size[{len(tgt_bbox)}]"
        # mask_loss = self.compute_Allmask_loss(src_mask,tgt_mask)
        
        
        match_idx = []
        matchness = []
        iou_ls = []
        for batch in range(N):
            nonOJ_all = False
            if tgt_category[batch].shape[0] == 0:
                nonOJ_all = True
                OJ_mask = torch.full([len(src_PC[batch])],False,device=src_PC[batch].device)
                match_idx += [{"OJ_mask":OJ_mask,'src_idx':None,'tgt_idx':None,"nonOJ_all":nonOJ_all}]
                continue
            
            
            batch_src_PC = src_PC[batch]

            batch_src_xcycwh = src_bbox[batch]
            # batch_src_xcycwh = xyloc2bbox(batch_src_PC,batch_src_xcycwh)
            
            batch_src_xyxy = bbox_xcychw2xyxy(batch_src_xcycwh)
            batch_tgt_xcycwh = tgt_bbox[batch]
            batch_tgt_xyxy = bbox_xcychw2xyxy(tgt_bbox[batch])
            EDLoss = self.EuDst(batch_src_PC,batch_tgt_xcycwh)

            Hungarian_result = [x for x in linear_sum_assignment(EDLoss.cpu().numpy())]
            Hungarian_result[0] = torch.tensor(Hungarian_result[0],device=src_PC[batch].device)
            Hungarian_result[1] = torch.tensor(Hungarian_result[1],device=src_PC[batch].device)

            assert (batch_src_xyxy[:, 2:] >= batch_src_xyxy[:, :2]).all()
            assert (batch_tgt_xyxy[:, 2:] >= batch_tgt_xyxy[:, :2]).all()


            iou,_ = box_iou(batch_src_xyxy,batch_tgt_xyxy)
            max_score,max_idx = torch.max(iou,dim=1)
            OJ_mask = max_score>self.iou_th
            OJ_mask[Hungarian_result[0]] = True
            # iou,OJ_mask,src_idx,tgt_idx = self.iou_assianment(batch_src_xyxy,batch_tgt_xyxy,iou,Hungarian_result)

            iou_ls += [iou]
            

            match_idx += [{"OJ_mask":OJ_mask,
                           'src_idx':Hungarian_result[0],
                           'tgt_idx':Hungarian_result[1],
                           "nonOJ_all":nonOJ_all}]
            matched = (iou[Hungarian_result[0],
                           Hungarian_result[1]])>self.iou_th
            # matchness += [torch.mean(1-diceloss[result[0],result[1]])]
            matchness += [torch.sum(matched)/max(len(matched),1)]

        return matchness,iou_ls,match_idx
    
    def iou_assianment(self,batch_src_xyxy,batch_tgt_xyxy,iou,Hungarian_result):
        src_arange = torch.arange(len(batch_src_xyxy),dtype=torch.long)
        
        max_score,max_idx = torch.max(iou,dim=1)
        OJ_mask = max_score>self.iou_th
        OJ_mask[Hungarian_result[0]] = True
        max_idx[Hungarian_result[0]] = Hungarian_result[1]
        
        src_idx = src_arange[OJ_mask]
        tgt_idx = max_idx[OJ_mask]
        
        return iou, OJ_mask, src_idx, tgt_idx
    def EuDst(self,src_xcyc,tgt_xcycwh):
        src_xc,src_yc = src_xcyc.unbind(dim=-1)
        tgt_xc,tgt_yc,tgt_w,tgt_h = tgt_xcycwh.unbind(dim=-1)
        
        src_xc = src_xc.unsqueeze(-1)
        src_yc = src_yc.unsqueeze(-1)
        
        EDLoss = ((src_xc-tgt_xc)**2+(src_yc-tgt_yc)**2)**0.5
        return EDLoss

    def MSE_loss(self,src_mask,tgt_mask):
        src_S,_,_ = src_mask.shape
        tgt_S,_,_ = tgt_mask.shape
        src_S = len(src_mask)
        S_src_mask = src_mask.flatten(-2,-1).unsqueeze(1).repeat(1,tgt_S,1)
        S_tgt_mask = tgt_mask.flatten(-2,-1).unsqueeze(0).repeat(src_S,1,1)
        
        MSEloss = torch.mean(self.MSE(S_src_mask,S_tgt_mask),dim=-1)

        return MSEloss        
        
    def focal_loss(self,inputs, targets, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        T_S,_,_ = targets.shape
        S_S,_,_ = inputs.shape
        src_mask = inputs[:,None].repeat(1,T_S,1,1)
        tgt_mask = targets[None].repeat(S_S,1,1,1)
        ce_loss = F.binary_cross_entropy(src_mask, tgt_mask, reduction="none")
        p_t = src_mask * tgt_mask + (1 - src_mask) * (1 - tgt_mask)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * tgt_mask + (1 - alpha) * (1 - tgt_mask)
            loss = alpha_t * loss
        
        loss = torch.mean(loss,(2,3))
        return loss
            
    def dice_loss(self,inputs, targets):
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
    
    def BCE_loss(self,inputs,targets):
        self.BCE_fn = torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='none')
        T_S,_,_ = targets.shape
        S_S,_,_ = inputs.shape
        src_mask = inputs[:,None].repeat(1,T_S,1,1).flatten(2,3)
        tgt_mask = targets[None].repeat(S_S,1,1,1).flatten(2,3)
        loss = self.BCE_fn(src_mask,tgt_mask)
        loss = torch.mean(loss,dim=(-1))
        return loss 
    def var_loss(self,inputs,targets):
        S_C,S_H,S_W = inputs.shape
        T_C,T_H,S_W = targets.shape
        SX_var = []
        SY_var= []
        TX_var = []
        TY_var= []
        
        for idx in range(S_C):
            S_X,S_Y = (inputs[idx]>0.4).nonzero(as_tuple=True)
            S_X = torch.unique(S_X)
            S_X = S_X.to(torch.float32).clone().detach().requires_grad_(True)/S_W
            S_Y = torch.unique(S_Y)
            S_Y = S_Y.to(torch.float32).clone().detach().requires_grad_(True)/S_H
            if len(S_X) == 0:
                SX_var += [torch.tensor(0,dtype=torch.float32,requires_grad=True).to(self.device)]
                SY_var += [torch.tensor(0,dtype=torch.float32,requires_grad=True).to(self.device)]
            else:
                S_Xmean,S_Ymean = S_X.mean(),S_Y.mean()
                SX_var += [((S_X*S_X).mean()-(S_Xmean*S_Xmean))]
                SY_var += [((S_Y*S_Y).mean()-(S_Ymean*S_Ymean))]
        SX_var = torch.stack(SX_var).unsqueeze(-1).repeat(1,T_C)
        SY_var = torch.stack(SY_var).unsqueeze(-1).repeat(1,T_C)
        
        for idx in range(T_C):
            T_X,T_Y = (targets[idx]>0.4).nonzero(as_tuple=True)
            T_X = torch.unique(T_X)
            T_X = T_X.to(torch.float32).clone().detach().requires_grad_(True)/S_W
            T_Y = torch.unique(T_Y)
            T_Y = T_Y.to(torch.float32).clone().detach().requires_grad_(True)/S_H
            T_Xmean,T_Ymean = T_X.mean(),T_Y.mean()
            TX_var += [((T_X*T_X).mean()-(T_Xmean*T_Xmean))]
            TY_var += [((T_Y*T_Y).mean()-(T_Ymean*T_Ymean))]
        TX_var = torch.stack(TX_var).unsqueeze(0).repeat(S_C,1)
        TY_var = torch.stack(TY_var).unsqueeze(0).repeat(S_C,1)
        TX_var_MSEloss = self.MSE(SX_var,TX_var)
        TY_var_MSEloss = self.MSE(SY_var,TY_var)
        var_MSEloss = torch.mean(torch.stack([TX_var_MSEloss,TY_var_MSEloss],dim=0),dim=0)
        return var_MSEloss
    def PPMCC_loss(self,src,tgt):
        src_mask = src.to(torch.float64)
        tgt_mask = tgt.to(torch.float64)
        S_S,S_H,S_W = src_mask.shape
        T_S,T_H,S_W = tgt_mask.shape
        src_mask = src_mask[:,None].repeat(1,T_S,1,1).flatten(2,3)
        tgt_mask = tgt_mask[None].repeat(S_S,1,1,1).flatten(2,3)

        src_mean = torch.mean(src_mask,dim=-1)
        tgt_mean = torch.mean(tgt_mask,dim=-1)
        
        src_std = (torch.mean(((src_mask**2)),dim=-1)-(src_mean**2))**0.5
        
        tgt_std = (torch.mean((tgt_mask**2),dim=-1)-\
                                      (tgt_mean**2))**0.5
        
        src_tgt_cov = (torch.mean((src_mask*tgt_mask),dim=-1)-(src_mean*tgt_mean))
        
        PPMCC = 1-((src_tgt_cov/(src_std*tgt_std))).to(torch.float32)
        return PPMCC
    def category_loss(self,src_category,tgt_category):
        S_S,S_C = src_category.shape
        T_S, = tgt_category.shape
        result = src_category[:,tgt_category]
        return result
            
class compute_segmentation_loss(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.MSE = nn.MSELoss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.device = device
        self.BCELoss = nn.BCELoss(reduction='none')
        self.classify_loss = compute_classify_loss()
        
    def forward(self,src,tgt,match_idx,dice_loss):
        '''
        Parameters
        ----------
        src : torch.tensor
            where dtype is float32 and image normalized already.
        mask : torch.tensor
            where dtype is float32 and image normalized already.

        Returns
        -------
        CE_loss : torch.tensor
            compute cross entropy loss result between src and mask.
        '''
        N = len(src)

        smoothL1loss = []
        # GIOULoss = []
        CIOULoss = []
        # confL1loss = []
        categoryloss = []

        for batch in range(N):
            src_PC = src[batch]["proposal_center"]
            src_bbox = src[batch]['bbox']
            # src_conf = src[batch]["conf"]
            src_category = src[batch]["category"]
            tgt_bbox = tgt[batch]['bbox']
            tgt_category = tgt[batch]["category"]
            if match_idx[batch]["nonOJ_all"] == False:
                CIOULoss += [self.CIOU_Loss(src_PC, src_bbox, tgt_bbox, match_idx[batch])]
                # GIOULoss += [self.GIOU_Loss(src_PC,src_bbox,tgt_bbox,match_idx[batch])]
                smoothL1loss += [self.L1_Loss(src_PC,src_bbox,tgt_bbox,match_idx[batch])]
            categoryloss += [self.classify_loss(src_category,tgt_category,match_idx[batch])]
                
            # confL1loss += [self.confL1_loss(src_conf, match_idx[batch])]

    
        # GIOULoss = torch.mean(torch.stack(GIOULoss,dim=0))
        CIOULoss = torch.mean(torch.stack(CIOULoss,dim=0))
        smoothL1loss = torch.mean(torch.stack(smoothL1loss,dim=0))
        categoryloss = torch.mean(torch.stack(categoryloss,dim=0))
        # confL1loss = torch.mean(torch.stack(confL1loss,dim=0))
        return {"CIOULoss":CIOULoss,
                "smoothL1loss":smoothL1loss,
                "categoryloss":categoryloss}
    
    def L1_Loss(self,src_PC,src_dxdywh,tgt_xcycwh,match_idx):

        # src_Px,src_Py = src_PC[match_idx["src_idx"]].unbind(dim=-1)
        src_xc,src_yc,src_w,src_h = src_dxdywh[match_idx["src_idx"]].unbind(-1)
        tgt_xc,tgt_yc,tgt_w,tgt_h = tgt_xcycwh[match_idx["tgt_idx"]].unbind(dim=-1)
        
        dx = (tgt_xc-src_xc)/src_w
        dy = (tgt_yc-src_yc)/src_h
        

        dw = torch.log(tgt_w/src_w)
        dh = torch.log(tgt_h/src_h)
        
        L1Loss = torch.stack([dx,dy,dw,dh],dim=-1)
        L1Loss = torch.mean(torch.sum(self.smooth_L1(L1Loss),dim=-1))
        return L1Loss

    def CIOU_Loss(self,src_PC, src_dxdywh, tgt_xcycwh, match_idx):
        
        src_xcycwh = src_dxdywh
        match_src_xcycwh = src_xcycwh[match_idx["src_idx"]]
        match_tgt_xcycwh = tgt_xcycwh[match_idx["tgt_idx"]]
        CIOUloss = CIoU(match_src_xcycwh,match_tgt_xcycwh)
        
        CIOUloss = torch.mean(CIOUloss)
        return CIOUloss

    def GIOU_Loss(self,src_PC,src_dxdywh,tgt_xcycwh,match_idx):
        src_xcycwh = xyloc2bbox(src_PC,src_dxdywh)
        match_src_bbox = bbox_xcychw2xyxy(src_xcycwh[match_idx["src_idx"]])
        match_tgt_bbox = bbox_xcychw2xyxy(tgt_xcycwh[match_idx["tgt_idx"]])
        GIOULoss = torch.diag(1-generalized_box_iou(match_src_bbox,match_tgt_bbox))
        
        GIOULoss = torch.mean(torch.sum(GIOULoss,dim=-1))
        
        return GIOULoss
    def bias_loss(self,src_bbox,src_bias,tgt_bbox,match_idx):
        sinf = torch.tensor([float("-inf")],device=src_bbox.device)
        inf = torch.tensor([float("inf")],device=src_bbox.device)

        eps = torch.finfo(src_bbox.dtype).eps
        match_src_bbox = src_bbox[match_idx["src_idx"]]
        match_src_bias = src_bias[match_idx["src_idx"]]
        match_tgt_bbox = tgt_bbox[match_idx["tgt_idx"]]

        tgt_bias = bbox2loc(match_src_bbox,match_tgt_bbox)
        
        biasloss = (match_src_bias-tgt_bias)
        
        nonmatch_mask = (torch.sum(biasloss == sinf,dim=-1)+torch.sum(biasloss == inf,dim=-1)+torch.sum(torch.isnan(biasloss),dim=-1))==0
        biasloss = biasloss[nonmatch_mask]
        biasloss = self.smooth_L1(biasloss)

        biasloss = torch.mean(torch.sum(self.smooth_L1(biasloss),dim=-1))
        
        return biasloss
    def CBAM_mask_loss(self,src_SA,tgt_mask):
        SAloss = self.MSE(src_SA.squeeze(),1-tgt_mask[-1])
        return torch.sum(SAloss)
    
    def MSE_loss(self,src_mask,tgt_mask,match_idx):
        S_src,H,W = src_mask.shape
        nonOJ_tgt_mask = torch.zeros([S_src,H,W],dtype=torch.float32,device=src_mask.device)
        MSEloss = 0
        if match_idx["nonOJ_all"]:
            MSEloss = torch.mean(torch.sum(self.MSE(src_mask,nonOJ_tgt_mask),dim=(-2,-1)))
            return MSEloss
        
        match_src_mask = src_mask[match_idx['src_idx']]
        match_tgt_mask = tgt_mask[match_idx['tgt_idx']]
        
        match_nonOJ_src_mask = src_mask[~match_idx["OJ_mask"]]
        match_nonOJ_tgt_mask = nonOJ_tgt_mask[~match_idx["OJ_mask"]]

        OJ_MSEloss = self.BCELoss(match_src_mask,match_tgt_mask)
        nonOJ_MSEloss = self.BCELoss(match_nonOJ_src_mask,match_nonOJ_tgt_mask)
        # MSEloss = torch.mean(torch.sum(OJ_MSEloss,dim=(-2,-1)))
        if nonOJ_MSEloss.shape[0] != 0:
            MSEloss += torch.mean(torch.sum(nonOJ_MSEloss,dim=(-2,-1)))
        if OJ_MSEloss.shape[0] != 0:
            MSEloss += torch.mean(torch.sum(OJ_MSEloss,dim=(-2,-1)))
        
        return MSEloss
        pass
    def smooth_L1(self,Loss):

        smoothL1loss = torch.where(torch.abs(Loss) < 1.,
                                   0.5 * Loss ** 2,
                                   torch.abs(Loss) - 0.5
                                   )

        return smoothL1loss
    def center_loss(self,src_mask,tgt_bbox,match_idx):
        if match_idx["nonOJ_all"]:
            return
        src_mask = src_mask[match_idx["src_idx"]]
        C,H,W = src_mask.shape
        cx,cy,h,w = torch.unbind(tgt_bbox[match_idx["tgt_idx"]],dim=-1)
        cx = (cx*H).to(torch.long)
        cy = (cy*W).to(torch.long)
        channelsele = torch.arange(0,C)
        
        eps = torch.finfo(src_mask.dtype).eps
        src_center = src_mask[channelsele,cy,cx].clamp(eps)
        
        centerloss = (-torch.log(src_center))

        
        centerloss = torch.mean(centerloss)
        
        return centerloss
    
    def hwbox_loss(self,src_bbox,tgt_bbox,match_idx):
        match_src_bbox = src_bbox[match_idx["src_idx"]][2:]
        match_tgt_bbox = tgt_bbox[match_idx["tgt_idx"]][2:]
        
        hwMSEloss = self.MSE(match_src_bbox,match_tgt_bbox)
        
        return hwMSEloss
    # def confL1_loss(self,src_conf,match_idx):

    #     if match_idx["nonOJ_all"]:
    #         nonOJ_src_conf = src_conf
    #         nonOJconfSL1loss = ((-torch.log(nonOJ_src_conf[:,0]))).sort(descending=True)[0]
            
    #         return torch.sum(nonOJconfSL1loss[:10])

    #     nonOJ_src_conf = src_conf[~match_idx["OJ_mask"],0].clamp(1e-5)
    #     OJ_src_conf = src_conf[match_idx["OJ_mask"],1].clamp(1e-5)
    #     confSL1loss = 0
    #     if OJ_src_conf.shape[0]!=0:
    #         confSL1loss += torch.mean((-torch.log(OJ_src_conf)))
            
    #     if nonOJ_src_conf.shape[0]!=0:
    #         confSL1loss += torch.mean((-torch.log(nonOJ_src_conf)))
        

    #     confSL1loss = confSL1loss/2

    #     return confSL1loss
    def confL1_loss(self,src_conf,match_idx):

        nonOJ_src_conf = src_conf[~match_idx["OJ_mask"],0].clamp(1e-5)
        OJ_src_conf = src_conf[match_idx["OJ_mask"],1].clamp(1e-5)
        OJ_len = max(len(OJ_src_conf),2)
        confSL1loss = 0
        
        if match_idx["nonOJ_all"]:
            nonOJ_src_conf = src_conf
            nonOJconfSL1loss = ((-torch.log(nonOJ_src_conf[:,0]))).sort(descending=True)[0]
            hard_nonOJ = nonOJconfSL1loss[:10]
            Other_nonOJ = nonOJconfSL1loss[10:]
            sampler = list(data.RandomSampler(Other_nonOJ))[:10]
            random_nonOJ = Other_nonOJ[sampler]
            
            nonOJconfSL1loss = torch.mean(torch.cat([hard_nonOJ,random_nonOJ],dim=-1))

            return nonOJconfSL1loss

        OJ_loss = (-torch.log(OJ_src_conf))
        nonOJ_loss = ((-torch.log(nonOJ_src_conf))).sort(descending=True)[0]
        if len(OJ_loss) != 0:
            confSL1loss += torch.mean(OJ_loss)
            
        if len(nonOJ_loss) != 0:
            nonOJ_hard_sample = nonOJ_loss[:OJ_len//2]
            nonOJ_Random_sample = nonOJ_loss[OJ_len//2:]
            nonOJ_Random_sample = nonOJ_Random_sample[list(data.RandomSampler(nonOJ_Random_sample))[:OJ_len//2]]
            
            confSL1loss += torch.mean(torch.cat([nonOJ_hard_sample,nonOJ_Random_sample],dim=-1))
        return confSL1loss
    def focal_loss(self,inputs, targets,match_idx, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        T_S,_,_ = targets.shape
        src_mask = inputs[match_idx['src_idx']]
        tgt_mask = targets[[match_idx['tgt_idx']]]
        ce_loss = F.binary_cross_entropy(src_mask, tgt_mask, reduction="none")
        p_t = src_mask * tgt_mask + (1 - src_mask) * (1 - tgt_mask)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * tgt_mask + (1 - alpha) * (1 - tgt_mask)
            loss = alpha_t * loss
        
        loss = torch.mean(loss,(-2,-1))
        return loss
            
    def dice_loss(self,inputs, targets,match_idx):
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
        num_object,_,_ = targets.shape
        src_mask = inputs[match_idx['src_idx']].flatten(-2,-1)
        tgt_mask = targets[[match_idx['tgt_idx']]].flatten(-2,-1)
        numerator = (2 * (src_mask * tgt_mask)).sum(-1)
        denominator = src_mask.sum(-1) + tgt_mask.sum(-1)
        loss = torch.sum(1 - (numerator + 1) / (denominator + 1))
        return loss
    def BCE_loss(self,inputs, targets,match_idx):
        src_mask = inputs[match_idx['src_idx']]
        tgt_mask = targets[[match_idx['tgt_idx']]]
        loss = nn.BCELoss(reduction='mean')(src_mask,tgt_mask)
        return loss 
    
    def var_loss(self,inputs,targets,match_idx):
        src_mask = inputs[match_idx['src_idx']]
        tgt_mask = targets[[match_idx['tgt_idx']]]
        S_C,S_H,S_W = src_mask.shape
        T_C,T_H,S_W = tgt_mask.shape
        SX_var = []
        SY_var= []
        TX_var = []
        TY_var= []
        for idx in range(S_C):
            S_X,S_Y = (src_mask[idx]>0.4).nonzero(as_tuple=True)
            S_X = torch.unique(S_X)
            S_X = S_X.to(torch.float32).clone().detach().requires_grad_(True)/S_W
            S_Y = torch.unique(S_Y)
            S_Y = S_Y.to(torch.float32).clone().detach().requires_grad_(True)/S_H
            if len(S_X) == 0:
                SX_var += [torch.tensor(0,dtype=torch.float32,requires_grad=True).to(self.device)]
                SY_var += [torch.tensor(0,dtype=torch.float32,requires_grad=True).to(self.device)]
            else:
                S_Xmean,S_Ymean = S_X.to(torch.float32).mean(),S_Y.to(torch.float32).mean()
                SX_var += [(S_X*S_X).mean()-(S_Xmean*S_Xmean)]
                SY_var += [(S_Y*S_Y).mean()-(S_Ymean*S_Ymean)]
        SX_var = torch.stack(SX_var)
        SY_var = torch.stack(SY_var)
        
        for idx in range(T_C):
            T_X,T_Y = (tgt_mask[idx]>0.4).nonzero(as_tuple=True)
            T_X = torch.unique(T_X)
            T_X = T_X.to(torch.float32).clone().detach().requires_grad_(True)/S_W
            T_Y = torch.unique(T_Y)
            T_Y = T_Y.to(torch.float32).clone().detach().requires_grad_(True)/S_W
            T_Xmean,T_Ymean = T_X.mean(),T_Y.mean()
            TX_var += [((T_X*T_X).mean()-(T_Xmean*T_Xmean))]
            TY_var += [((T_Y*T_Y).mean()-(T_Ymean*T_Ymean))]
        TX_var = torch.stack(TX_var)
        TY_var = torch.stack(TY_var)
        TX_var_MSEloss = self.MSE(SX_var,TX_var)
        TY_var_MSEloss = self.MSE(SY_var,TY_var)
        
        var_MSEloss = torch.mean(torch.stack([TX_var_MSEloss,TY_var_MSEloss],dim=0))
        return var_MSEloss
    def PPMCC_loss(self,src,tgt,match_idx):
        src_mask = src.to(torch.float64)[match_idx['src_idx']]
        tgt_mask = tgt.to(torch.float64)[match_idx['tgt_idx']]
        
        src_mean = torch.mean(src_mask,dim=-1,keepdim=True)
        tgt_mean = torch.mean(tgt_mask,dim=-1,keepdim=True)
        
        src_std = (torch.mean(((src_mask**2)),dim=-1,keepdim=True)-(src_mean**2))**0.5
        
        tgt_std = (torch.mean((tgt_mask**2),dim=-1,keepdim=True)-\
                                      (tgt_mean**2))**0.5
        
        src_tgt_cov = (torch.mean((src_mask*tgt_mask),dim=-1,keepdim=True)-(src_mean*tgt_mean))
        PPMCC = torch.mean(1-(src_tgt_cov/(src_std*tgt_std))).to(torch.float32)
        return PPMCC
class compute_classify_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.nonOJ_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self,src_category,tgt_category,match_idx):

            
        categoryloss = self.compute_category_loss(src_category,tgt_category,match_idx)
        return categoryloss    
        
    def compute_category_loss(self,src_category,tgt_category,match_idx):
        # pdb.set_trace()
        nonOJ_src_category = src_category[~match_idx['OJ_mask']]
        nonOJ_tgt_category = torch.full([len(nonOJ_src_category)],6,dtype=torch.long,device=nonOJ_src_category.device)

        nonOJ_loss = self.cross_entropy(nonOJ_src_category,nonOJ_tgt_category).sort(descending=True)[0]

        if match_idx["nonOJ_all"]:
            nonOJ_hard_sample = nonOJ_loss[:5]
            nonOJ_random_sample = nonOJ_loss[10//2:][list(data.RandomSampler(nonOJ_loss[10//2:]))[:10//2]]
            
            nonOJ_loss = torch.cat([nonOJ_hard_sample,nonOJ_random_sample],dim=-1)
            return torch.mean(nonOJ_loss)
       
        match_src_category = src_category[match_idx['src_idx']]
        match_tgt_category = tgt_category[match_idx['tgt_idx']]
                
        OJ_loss = self.cross_entropy(match_src_category,match_tgt_category)
        OJ_num = max((len(OJ_loss)//2),1)
        nonOJ_hard_sample = nonOJ_loss[:OJ_num]
        if nonOJ_loss[OJ_num:].shape[0] != 0:
            nonOJ_random_sample = nonOJ_loss[OJ_num:][list(data.RandomSampler(nonOJ_loss[OJ_num:]))[:OJ_num]]
            nonOJ_loss = torch.cat([nonOJ_hard_sample,nonOJ_random_sample],dim=-1)
        else:
            nonOJ_loss = nonOJ_hard_sample
        batch_loss = torch.mean(OJ_loss)+torch.mean(nonOJ_loss)

        return batch_loss

@torch.no_grad()
def confusion_matrix_F(src,tgt,num_object,match_idx):
    
    confusion_matrix = np.zeros([num_object+1,num_object+1])
    N = len(src)
    soft_max = torch.nn.Softmax(dim=-1)
    for batch in range(N):
        soft_max_src = soft_max(src[batch]['category'])
        match_src_category = soft_max_src[match_idx[batch]['src_idx']]
        if tgt[batch]['category'].shape[0] == 0:
            continue

        match_tgt_category = tgt[batch]['category'][match_idx[batch]['tgt_idx']]

        S,_ = match_src_category.shape
        if S > 0:
            max_category = torch.argmax(match_src_category,dim=-1)
            
            for i in range(len(match_tgt_category)):
                confusion_matrix[int(match_tgt_category[i].item()),
                                 int(max_category[i].item())] +=1
    return confusion_matrix
@torch.no_grad()   
def Conf_confusion_matrix_F(src,tgt,match_idx,diceloss_ls):
    confusion_matrix = np.zeros([2,2])
    suppression_matrix = np.zeros([2,2])
    N = len(src)
    for batch in range(N):
        src_conf = src[batch]['conf']
        OJ_mask = match_idx[batch]["OJ_mask"]

        nonOJ_src = src_conf[~OJ_mask]
        if nonOJ_src.shape[0] != 0:
            nonOJ_src_idx = torch.argmax(nonOJ_src,dim=-1)
            nonOJ_src_tgt_mask = nonOJ_src_idx==0
            nonOJ_PP = sum(nonOJ_src_tgt_mask)
            nonOJ_NP = sum(~nonOJ_src_tgt_mask)
            suppression_matrix[1,1] += nonOJ_PP
            suppression_matrix[0,1] += nonOJ_NP
            
        OJ_src = src_conf[OJ_mask]
        if OJ_src.shape[0] != 0:
            OJ_src_idx = torch.argmax(OJ_src,dim=-1)
            OJ_src_tgt_mask = OJ_src_idx == 1
            OJ_PP = sum(OJ_src_tgt_mask)
            OJ_NP = sum(~OJ_src_tgt_mask)
            confusion_matrix[1,1] += OJ_PP
            confusion_matrix[0,1] += OJ_NP

        
        

    return suppression_matrix,confusion_matrix