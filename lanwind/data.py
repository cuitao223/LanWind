"""Dataset adapter for user-owned, explicitly split local input bundles."""
from pathlib import Path
import numpy as np
import torch
from .io import read_bundle,features

def sample_paths(directory):
    paths=sorted(Path(directory).glob('*.npz'))
    if not paths:raise FileNotFoundError('No local .npz input bundles found in '+str(directory))
    return paths

def targets(data,cfg,baseline_valid):
    future=np.asarray(data['future_uv_ms'],dtype='float32');times=np.asarray(data['future_times_utc'],dtype='int64');origin=data['history_times_utc'][-1]
    steps=cfg['model']['forecast_steps'];z=cfg['model']['levels'];h,w=cfg['grid']['shape']
    if future.ndim!=5 or future.shape[1:]!=(2,z,h,w) or not 1<=len(future)<=steps or times.shape!=(len(future),):raise ValueError('Future target shape mismatch')
    gaps=np.diff(np.r_[origin,times])
    if np.any(gaps<=0) or np.any(gaps>cfg['data']['max_gap_seconds']):raise ValueError('Targets cross a long gap or have invalid times')
    if times[-1]-origin>cfg['data']['forecast_minutes']*60+cfg['data']['max_gap_seconds']:raise ValueError('Targets exceed bracketing forecast scope')
    mask=np.isfinite(future).all(1)&baseline_valid[None];mean=np.asarray(cfg['normalization']['mean_uv'])[None,:,None,None,None];std=np.asarray(cfg['normalization']['std_uv'])[None,:,None,None,None]
    y=np.zeros((steps,2*z,h,w),dtype='float32');valid=np.zeros((steps,z,h,w),dtype=bool)
    y[:len(future)]=np.where(mask[:,None],(future-mean)/std,0).reshape(len(future),2*z,h,w);valid[:len(future)]=mask
    if not valid.any():raise ValueError('No valid supervised targets')
    return y,valid

class LocalBundles(torch.utils.data.Dataset):
    def __init__(self,paths,cfg):self.paths=list(paths);self.cfg=cfg
    def __len__(self):return len(self.paths)
    def __getitem__(self,index):
        data=read_bundle(self.paths[index]);x,valid=features(data,self.cfg);y,mask=targets(data,self.cfg,valid)
        return {**x,'target':y,'target_mask':mask}

def verify_partitions(train,val):
    # Complete history-through-target spans may not overlap between partitions.
    def bounds(paths):
        low=np.iinfo(np.int64).max;high=0
        for p in paths:
            data=read_bundle(p);low=min(low,int(data['history_times_utc'][0]));high=max(high,int(data['future_times_utc'][-1]))
        return low,high
    tr,va=bounds(train),bounds(val)
    if tr[1]>=va[0]:raise ValueError('Training and validation windows overlap in time')
    return {'train_utc_bounds':tr,'validation_utc_bounds':va,'test_accessed':False}

def fit_training_normalization(paths,cfg):
    total=np.zeros(2,dtype='float64');squares=np.zeros(2,dtype='float64');count=0;dt_sum=0.;dt_count=0
    for path in paths:
        data=read_bundle(path)
        for key in ['history_uv_ms','future_uv_ms']:
            a=np.asarray(data[key]);valid=np.isfinite(a).all(1)
            for c in range(2):
                v=a[:,c][valid].astype('float64');total[c]+=v.sum();squares[c]+=(v*v).sum()
            count+=int(valid.sum())
        observed=np.isfinite(data['history_uv_ms']).all(1);previous=np.full(observed.shape[1:],np.nan)
        for time,mask in zip(data['history_times_utc'],observed):
            elapsed=time-previous;ok=mask&np.isfinite(previous)&(elapsed>0)&(elapsed<=cfg['data']['max_gap_seconds'])
            dt_sum+=float(elapsed[ok].sum()/60);dt_count+=int(ok.sum());previous=np.where(mask,time,previous)
    if count==0 or dt_count==0:raise ValueError('Insufficient training observations for normalization')
    mean=total/count;std=np.sqrt(np.maximum(squares/count-mean*mean,1e-8))
    cfg['normalization']={'mean_uv':mean.tolist(),'std_uv':std.tolist(),'dt_mean_minutes':dt_sum/dt_count,'dt_missing_value':-1.,'fitted_on':'user_training_partition_only'}
    cfg['risk']['threshold_s_1']=None
    return cfg
