"""Train one LanWind instance from chronological, user-owned local bundles."""
import argparse,csv,json,random,sys,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from lanwind import LanWind
from lanwind.io import config
from lanwind.data import sample_paths,LocalBundles,verify_partitions,fit_training_normalization
from lanwind.model import masked_sequence_loss

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True,help='Local directory containing train/ and val/');p.add_argument('--output',required=True);p.add_argument('--config');p.add_argument('--batch-size',type=int);p.add_argument('--device',default='cuda');a=p.parse_args()
    if not a.device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('Run training on a CUDA server; this command never falls back to CPU')
    out=Path(a.output)
    if out.exists():raise FileExistsError('Choose a new run directory to preserve previous runs')
    cfg=copy.deepcopy(config(a.config));settings=cfg['training'];batch=a.batch_size or settings['batch_size']
    if batch<=0:raise ValueError('Batch size must be positive')
    train=sample_paths(Path(a.data)/'train');val=sample_paths(Path(a.data)/'val');split=verify_partitions(train,val)
    cfg=fit_training_normalization(train,cfg);out.mkdir(parents=True);(out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf8');(out/'partition_audit.json').write_text(json.dumps(split,indent=2),encoding='utf8')
    seed=settings['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.set_num_threads(4)
    loaders={name:torch.utils.data.DataLoader(LocalBundles(paths,cfg),batch_size=batch,shuffle=name=='train',num_workers=settings['workers'],pin_memory=True) for name,paths in [('train',train),('val',val)]}
    model=LanWind(levels=cfg['model']['levels'],forecast_steps=cfg['model']['forecast_steps']).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    best=float('inf');stale=0;start=time.monotonic()
    with (out/'training.csv').open('w',newline='',encoding='utf8') as log:
        writer=csv.DictWriter(log,fieldnames=['epoch','train_loss','val_loss','seconds']);writer.writeheader()
        for epoch in range(1,settings['epochs']+1):
            losses={}
            for name,loader in loaders.items():
                model.train(name=='train');numerator=0.;denominator=0
                for batch_data in loader:
                    b={k:v.to(a.device,non_blocking=True) for k,v in batch_data.items()}
                    with torch.set_grad_enabled(name=='train'):
                        result=model(**{k:b[k] for k in ['history','history_lengths','baseline','nominal_lead_seconds','gfs']});loss=masked_sequence_loss(result,b['target'],b['target_mask'])
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                        if name=='train':opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                    n=int(b['target_mask'].sum())*2;numerator+=float(loss.detach())*n;denominator+=n
                losses[name]=numerator/denominator
            record={'epoch':epoch,'train_loss':losses['train'],'val_loss':losses['val'],'seconds':time.monotonic()-start};writer.writerow(record);log.flush();print(json.dumps(record),flush=True)
            if losses['val']<best:
                best=losses['val'];stale=0;torch.save(model.state_dict(),out/'best.pt');(out/'best.json').write_text(json.dumps({**record,'batch_size':batch,'selection':'validation_only'},indent=2),encoding='utf8')
            else:stale+=1
            if stale>=settings['patience']:break
    print('TRAINING_COMPLETE',out,'test partition was not accessed')

if __name__=='__main__':main()
