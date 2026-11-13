"""Local-only physical-unit input validation and causal feature construction."""
from pathlib import Path
import numpy as np
import torch
from . import LanWind
from .configuration import config

ROOT=Path(__file__).resolve().parents[1]

def read_bundle(path):
    with np.load(path,allow_pickle=False) as data:
        return {key:np.array(data[key]) for key in data.files}

def features(data,cfg):
    z=cfg['model']['levels'];steps=cfg['model']['forecast_steps'];maximum=cfg['model']['maximum_input_frames']
    history=np.asarray(data['history_uv_ms'],dtype='float32')
    times=np.asarray(data['history_times_utc'],dtype='int64')
    if history.ndim!=5 or history.shape[1:3]!=(2,z) or times.shape!=(len(history),):
        raise ValueError('history_uv_ms must be [T,2,Z,Y,X], with T UTC timestamps')
    if not 2<=len(times)<=maximum or np.any(np.diff(times)<=0):
        raise ValueError('Supply 2..23 chronological history frames; duplicates are invalid')
    if history.shape[-2:]!=tuple(cfg['grid']['shape']):raise ValueError('Spatial grid must match config')
    lookback=cfg['data']['history_minutes']*60
    if np.any(np.diff(times)>cfg['data']['max_gap_seconds']) or times[-1]-times[0]>lookback+cfg['data']['max_gap_seconds']:
        raise ValueError('History crosses a long gap or exceeds the bracketing lookback')
    if times[0]<times[-1]-lookback and times[1]<=times[-1]-lookback:
        raise ValueError('At most one history scan may bracket the two-hour boundary')
    if np.isinf(history).any():raise ValueError('Use NaN, not Inf, for missing observations')
    cadence=float(np.median(np.diff(times)));nominal=np.arange(1,steps+1,dtype='float32')*cadence
    gfs=np.asarray(data['gfs_uv_ms'],dtype='float32')
    if gfs.shape!=(steps,2,*history.shape[-2:]) or not np.isfinite(gfs).all():
        raise ValueError('GFS must be finite [60,2,Y,X], aligned with the decoder clock')
    leads=np.asarray(data['gfs_lead_seconds'])
    if leads.shape!=nominal.shape or not np.allclose(leads,nominal,atol=.01,rtol=0):
        raise ValueError('GFS leads must equal step index times historical median cadence')
    if np.asarray(data['gfs_available_at_utc']).size!=1 or int(np.asarray(data['gfs_available_at_utc']).item())>times[-1]:
        raise ValueError('GFS must already be available at the forecast origin')
    mean=np.asarray(cfg['normalization']['mean_uv'],dtype='float32')[None,:,None,None,None]
    std=np.asarray(cfg['normalization']['std_uv'],dtype='float32')[None,:,None,None,None]
    if np.any(std<=0):raise ValueError('Normalization standard deviations must be positive')
    observed=np.isfinite(history).all(1);normalized=np.where(observed[:,None],(history-mean)/std,0)
    previous=np.full(observed.shape[1:],np.nan);delta=np.full(observed.shape,-1,dtype='float32');base=np.zeros(history.shape[1:],dtype='float32');valid=np.zeros(observed.shape[1:],dtype=bool)
    for i,mask in enumerate(observed):
        elapsed=times[i]-previous;known=mask & np.isfinite(previous) & (elapsed>0) & (elapsed<=cfg['data']['max_gap_seconds'])
        delta[i]=np.where(known,elapsed/60/cfg['normalization']['dt_mean_minutes'],-1)
        previous=np.where(mask,times[i],previous);base=np.where(mask[None],normalized[i],base);valid|=mask
    if not valid.any():raise ValueError('History contains no valid paired u/v observations')
    n=len(history);h,w=history.shape[-2:];x=np.zeros((maximum,4*z,h,w),dtype='float32')
    x[-n:]=np.concatenate([normalized.reshape(n,2*z,h,w),observed.astype('float32'),delta],axis=1)
    forcing=(gfs-mean[:, :, 0])/std[:, :, 0]
    return dict(history=x,history_lengths=np.int64(n),baseline=base.reshape(2*z,h,w),nominal_lead_seconds=nominal,gfs=forcing.astype('float32')),valid

def load_model(cfg=None,weight=None,device='cpu'):
    cfg=cfg or config();model=LanWind(levels=cfg['model']['levels'],forecast_steps=cfg['model']['forecast_steps'],use_gfs=True)
    checkpoint=torch.load(weight or ROOT/'weights/lanwind.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint.get('model_state',checkpoint),strict=True)
    return model.to(device).eval()

def predict(data,cfg=None,weight=None,device='cpu',model=None):
    cfg=cfg or config();inputs,mask=features(data,cfg);model=model or load_model(cfg,weight,device)
    batch={k:torch.as_tensor(v,device=device).unsqueeze(0) for k,v in inputs.items()}
    with torch.inference_mode():result=model(**batch)[0].cpu().numpy()
    z=cfg['model']['levels'];result=result.reshape(len(result),2,z,*result.shape[-2:])
    mean=np.asarray(cfg['normalization']['mean_uv'])[None,:,None,None,None];std=np.asarray(cfg['normalization']['std_uv'])[None,:,None,None,None]
    result=(result*std+mean).astype('float32')
    if not np.isfinite(result).all():raise FloatingPointError('Model output contains NaN or Inf')
    baseline=inputs['baseline'].reshape(2,z,*result.shape[-2:])*std[0]+mean[0]
    threshold=cfg['risk']['threshold_s_1']
    return dict(prediction_uv_ms=result,baseline_uv_ms=baseline.astype('float32'),baseline_valid=mask,nominal_lead_seconds=inputs['nominal_lead_seconds'],origin_utc=np.int64(data['history_times_utc'][-1]),risk_threshold_s_1=np.float64(threshold if threshold is not None else np.nan),grid_spacing_m=np.float64(cfg['grid']['spacing_m']))
