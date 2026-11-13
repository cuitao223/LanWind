"""Validate checkpoint loading and model execution using in-memory synthetic tensors."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from lanwind.io import config,predict,features

def main():
    torch.set_num_threads(2);cfg=config();rng=np.random.default_rng(42)
    times=np.int64(1700000000)+np.arange(4)*367
    data=dict(history_uv_ms=rng.normal(size=(4,2,5,33,33)).astype('float32'),history_times_utc=times,gfs_uv_ms=np.zeros((60,2,33,33),dtype='float32'),gfs_lead_seconds=np.arange(1,61)*367,gfs_available_at_utc=times[-1]-21600)
    data['history_uv_ms'][0,:,:,0,0]=np.nan
    result=predict(data,cfg);assert result['prediction_uv_ms'].shape==(60,2,5,33,33);assert np.isfinite(result['prediction_uv_ms']).all()
    for field,value in [('gfs_available_at_utc',times[-1]+1),('history_times_utc',times[::-1])]:
        invalid={**data,field:value}
        try:features(invalid,cfg)
        except ValueError:pass
        else:raise AssertionError('Expected a causal-input validation failure')
    print('SMOKE_TEST_PASSED: released weights loaded, finite 60-step output, causal validation passed; synthetic tensors only; no files generated')

if __name__=='__main__':main()
