"""Evaluate local test bundles once with persistence and the selected LanWind model."""
import argparse,json,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from lanwind.io import config,read_bundle,predict,load_model
from lanwind.data import sample_paths,targets
from lanwind.metrics import wind_direction,horizontal_shear

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True,help='Local test bundle directory');p.add_argument('--output',required=True);p.add_argument('--config');p.add_argument('--weights');p.add_argument('--device',default='cpu');a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError('Existing evaluation preserved; choose a new path')
    cfg=config(a.config);torch.set_num_threads(4);model=load_model(cfg,a.weights,a.device);acc={};calm=cfg['evaluation']['calm_speed_ms'];threshold=cfg['risk']['threshold_s_1'];count=0
    for path in sample_paths(a.data):
        data=read_bundle(path);result=predict(data,cfg,device=a.device,model=model);targets(data,cfg,result['baseline_valid']);truth=np.asarray(data['future_uv_ms']);times=np.asarray(data['future_times_utc'])-data['history_times_utc'][-1]
        for minutes in cfg['evaluation']['horizons_minutes']:
            i=int(np.argmin(abs(times-minutes*60)))
            if abs(times[i]-minutes*60)>cfg['evaluation']['nearest_scan_tolerance_seconds']:continue
            real=truth[i];mask=np.isfinite(real).all(0)&result['baseline_valid'];comparison={'lanwind':result['prediction_uv_ms'][i],'persistence':result['baseline_uv_ms']};actual_speed=np.hypot(real[0],real[1]);direction_mask=mask&(actual_speed>=calm)
            for pred in comparison.values():direction_mask&=np.hypot(pred[0],pred[1])>=calm
            for name,pred in comparison.items():
                r=acc.setdefault(str(minutes),{}).setdefault(name,dict(squared=0.,speed_error=0.,direction_error=0.,valid_cells=0,direction_valid_cells=0,true_calm_cells=0,risk_tp=0,risk_fp=0,risk_fn=0,risk_positives=0))
                diff=pred.astype('float64')-real;angular=abs((wind_direction(pred[0],pred[1])-wind_direction(real[0],real[1])+180)%360-180)
                r['squared']+=float((diff[0]**2+diff[1]**2)[mask].sum());r['speed_error']+=float(abs(np.hypot(pred[0],pred[1])-actual_speed)[mask].sum(dtype='float64'));r['direction_error']+=float(angular[direction_mask].sum(dtype='float64'));r['valid_cells']+=int(mask.sum());r['direction_valid_cells']+=int(direction_mask.sum());r['true_calm_cells']+=int((mask&(actual_speed<calm)).sum())
                if threshold is not None:
                    real_shear=horizontal_shear(real,mask,cfg['grid']['spacing_m']);pred_shear=horizontal_shear(pred,mask,cfg['grid']['spacing_m']);valid=np.isfinite(real_shear);positive=valid&(real_shear>threshold);detected=valid&(pred_shear>threshold);r['risk_tp']+=int((positive&detected).sum());r['risk_fp']+=int((~positive&detected).sum());r['risk_fn']+=int((positive&~detected).sum());r['risk_positives']+=int(positive.sum())
        count+=1
    for models in acc.values():
        for r in models.values():
            r['vector_rmse_ms']=math.sqrt(r.pop('squared')/r['valid_cells']) if r['valid_cells'] else None;r['speed_mae_ms']=r.pop('speed_error')/r['valid_cells'] if r['valid_cells'] else None;r['direction_circular_mae_deg']=r.pop('direction_error')/r['direction_valid_cells'] if r['direction_valid_cells'] else None;den=2*r['risk_tp']+r['risk_fp']+r['risk_fn'];r['risk_f1']=2*r['risk_tp']/den if den and threshold is not None else None
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'samples':count,'comparison':'LanWind and persistence; common non-calm direction mask for these two objects','metrics':acc},indent=2),encoding='utf8');print('EVALUATION_SAVED',out)

if __name__=='__main__':main()
