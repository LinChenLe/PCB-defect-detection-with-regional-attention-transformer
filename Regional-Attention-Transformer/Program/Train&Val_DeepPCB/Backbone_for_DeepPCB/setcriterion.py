from torch import nn
import torch
from utils import bbox_xcychw2xyxy,bbox_area
from scipy.optimize import linear_sum_assignment
import pdb
import numpy as np

class Hungarian_match(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device=device
        self.softmax = nn.Softmax(dim=-1)
        self.MSE = nn.MSELoss(reduction='none')
    @torch.no_grad()
    # Hungarian(segmen_result,mask)
    def forward(self,src,tgt):
        '''
        src = {'mask': for each defect, 'category': [probability category]}
        tgt = ['mask': for each defect, 'category': [category]}
        '''
        
        N = len(src)
        src_mask = [x['mask'] for x in src]
        tgt_mask = [x['mask'] for x in tgt]
        mask_loss = self.compute_Allmask_loss(src_mask,tgt_mask)
        match_idx = []
        for batch in range(N):

            result = linear_sum_assignment(mask_loss[batch].cpu().numpy())
            match_idx += [{'src_idx':result[0],'tgt_idx':result[1]}]
        return mask_loss,match_idx
    def compute_Allmask_loss(self,src_mask,tgt_mask):
        N = len(src_mask)
        batch_loss = []
        for batch in range(N):

            src_S = len(src_mask[batch])
            S_src_mask = src_mask[batch].flatten(-2,-1)
            S_tgt_mask = tgt_mask[batch].repeat(src_S,1,1,1).permute(1,0,2,3).flatten(-2,-1)
            tgt_S,_,_ = S_tgt_mask.shape
            
            seq_loss = []
            for tgt_seq in range(tgt_S):
                seq_loss += [torch.sum(self.MSE(S_src_mask,S_tgt_mask[tgt_seq]),dim=-1)]
            batch_loss += [torch.stack(seq_loss,dim=-1)]
        return batch_loss        
        
        
class compute_backbone_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.softmax = nn.Softmax(dim=-1)
        self.SSE = nn.MSELoss(reduction="none")
    def forward(self,src,categories):
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
        loss = torch.sum(self.cross_entropy(src,categories))
        return {'category_loss':loss}

class compute_classify_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.nonOJ_cross_entropy = nn.CrossEntropyLoss(reduction='none')
    def forward(self,src,tgt,match_idx):
        src_category = [x['category'] for x in src]
        tgt_category = [x['category'] for x in tgt]
        category_loss = self.compute_category_loss(src_category,tgt_category,match_idx)
        return {'category_loss':category_loss}
    def compute_category_loss(self,src_category,tgt_category,match_idx):
        N = len(src_category)
        batch_loss = []
        for batch in range(N):
            S_src,_ = src_category[batch].shape
            S_tgt, = tgt_category[batch].shape
            nonOJ_category = torch.tensor([6],dtype=torch.long,device='cuda:0')
            tgt_category[batch] = torch.cat([tgt_category[batch],nonOJ_category])
            src_nonOJ_idx = np.arange(0,S_src)
            tgt_nonOJ_idx = np.full([S_src],-1)
            nonOJ_mask = np.full([S_src],True)
            nonOJ_mask[match_idx[batch]['src_idx']] = False
            src_nonOJ_idx = src_nonOJ_idx[nonOJ_mask]
            tgt_nonOJ_idx = tgt_nonOJ_idx[nonOJ_mask]
            
            
            
            
            match_src_category = src_category[batch][match_idx[batch]['src_idx']]
            match_tgt_category = tgt_category[batch][match_idx[batch]['tgt_idx']]
            
            nonOJ_src_category = src_category[batch][src_nonOJ_idx]
            nonOJ_tgt_category = tgt_category[batch][tgt_nonOJ_idx]
            nonOJ_loss = torch.mean(self.nonOJ_cross_entropy(nonOJ_src_category,nonOJ_tgt_category).sort()[0][:S_tgt])
            batch_loss += [self.cross_entropy(match_src_category,match_tgt_category)+nonOJ_loss]
            
        batch_loss = torch.sum(torch.stack(batch_loss))
        return batch_loss
        pass

@torch.no_grad()
def confusion_matrix_F(src,tgt,num_object):
    
    confusion_matrix = np.zeros([num_object+1,num_object+1])
    N = len(src)
    soft_max = torch.nn.Softmax(dim=-1)
    for batch in range(N):
        if tgt[batch]==6:
            continue
        soft_max_src = soft_max(src[batch])
        max_category = torch.argmax(soft_max_src,dim=-1)
        confusion_matrix[int(tgt[batch].item()),
                         int(max_category.item())] +=1

    return confusion_matrix