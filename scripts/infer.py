"""Run the released weights against a local input bundle; no data is distributed."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from lanwind.io import config,read_bundle,predict

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--config');p.add_argument('--weights');p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(4);out=Path(a.output)
    if out.exists():raise FileExistsError('Choose a new output filename')
    data=read_bundle(a.input);result=predict(data,config(a.config),a.weights,a.device)
    for key in ['future_uv_ms','future_times_utc']:
        if key in data:result[key]=data[key]
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**result)
    print('PREDICTION_SAVED',out,'shape',result['prediction_uv_ms'].shape)

if __name__=='__main__':main()
