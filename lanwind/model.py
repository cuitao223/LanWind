'''One two-layer residual ConvLSTM family, with optional known GFS forcing.'''
import torch
from torch import nn
from torch.nn import functional as F

from .cells import ConvLSTMCell


class SixHourConvLSTM(nn.Module):
    def __init__(self, levels=5, forecast_steps=60, use_gfs=True):
        super().__init__()
        self.levels = levels
        self.forecast_steps = forecast_steps
        self.use_gfs = use_gfs
        self.enc1 = ConvLSTMCell(4*levels, 64)
        self.enc2 = ConvLSTMCell(64, 32)
        self.dec1 = ConvLSTMCell(32, 64)
        self.dec2 = ConvLSTMCell(64, 32)
        self.lead_embeddings = nn.Parameter(torch.zeros(forecast_steps,32,1,1))
        self.time_projection = nn.Linear(1,32)
        self.gfs_projection = nn.Conv2d(2,32,1) if use_gfs else None
        self.head = nn.Conv2d(32,2*levels,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, history, history_lengths, baseline, nominal_lead_seconds, gfs=None):
        if history.ndim != 5 or history.shape[2] != 4*self.levels:
            raise ValueError('Expected history B,T,4Z,Y,X')
        batch, steps, _, height, width = history.shape
        if baseline.shape != (batch,2*self.levels,height,width):
            raise ValueError('Latest-valid residual baseline shape mismatch')
        if nominal_lead_seconds.shape != (batch,self.forecast_steps):
            raise ValueError('Nominal lead clock must align with decoder steps')
        if torch.any(history_lengths < 1) or torch.any(history_lengths > steps):
            raise ValueError('Invalid history lengths')
        if self.use_gfs and (gfs is None or gfs.shape != (batch,self.forecast_steps,2,height,width)):
            raise ValueError('GFS must be B,L,2,Y,X and known at forecast origin')
        a = tuple(history.new_zeros(batch,64,height,width) for _ in range(2))
        b = tuple(history.new_zeros(batch,32,height,width) for _ in range(2))
        for index in range(steps):
            active = (index >= steps-history_lengths).view(batch,1,1,1)
            candidate_a = self.enc1(history[:,index],a)
            a = tuple(torch.where(active,new,old) for new,old in zip(candidate_a,a))
            candidate_b = self.enc2(a[0],b)
            b = tuple(torch.where(active,new,old) for new,old in zip(candidate_b,b))
        context = b[0]; outputs = []
        for index in range(self.forecast_steps):
            lead = nominal_lead_seconds[:,index:index+1]/21600.
            forcing = context+self.lead_embeddings[index]+self.time_projection(lead)[:,:,None,None]
            if self.use_gfs:
                forcing = forcing+self.gfs_projection(gfs[:,index])
            a = self.dec1(forcing,a)
            b = self.dec2(a[0],b)
            outputs.append(self.head(b[0])+baseline)
        return torch.stack(outputs,1)


def masked_sequence_loss(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != (prediction.shape[0],prediction.shape[1],prediction.shape[2]//2,*prediction.shape[3:]):
        raise ValueError('Sequence target/mask dimensions do not align')
    expanded = mask.repeat(1,1,2,1,1)
    if not expanded.any():
        raise ValueError('No valid target cells in batch')
    safe_prediction = torch.where(expanded,prediction,0.)
    safe_target = torch.where(expanded,target,0.)
    values = F.smooth_l1_loss(safe_prediction,safe_target,reduction='none')
    return values.sum()/expanded.sum()
