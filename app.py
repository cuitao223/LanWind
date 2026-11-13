"""Local dashboard: no observations or replay bundles are shipped with the repository."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from lanwind.configuration import config
from lanwind.metrics import horizontal_shear

st.set_page_config(page_title='澜风 LanWind',layout='wide')
st.title('澜风 LanWind')
st.caption('Residual ConvLSTM · radar + GFS · six-hour wind-field forecasting')
cfg=config();path=st.sidebar.text_input('Local prediction file / 本机预测文件',placeholder='runs/prediction.npz')
st.sidebar.caption('Files stay on the machine running Streamlit. No dataset is bundled.')
if not path:
    st.info('先运行 scripts/infer.py，再填写本机输出文件路径。Run local inference, then enter its output path.')
    st.image('assets/architecture.png',caption='Background-conditioned residual ConvLSTM')
    st.stop()
p=Path(path).expanduser()
if not p.is_file():st.warning('Local file not found / 找不到本机文件');st.stop()
with np.load(p,allow_pickle=False) as f:data={k:np.array(f[k]) for k in f.files}
prediction=data['prediction_uv_ms'];z=st.sidebar.selectbox('Height relative to radar / 相对雷达高度',range(5),format_func=lambda i:f'{cfg["grid"]["heights_m"][i]} m');step=st.sidebar.slider('Forecast step / 预测步',0,len(prediction)-1,min(9,len(prediction)-1));mask=data['baseline_valid'][z]
st.caption(f'Historical origin / 历史起报时刻 (UTC): {pd.to_datetime(int(data["origin_utc"]),unit="s",utc=True)} · nominal lead: {float(data["nominal_lead_seconds"][step])/60:.1f} min')
cols=st.columns(2);fields=[('Latest valid history',data['baseline_uv_ms']),('LanWind forecast',prediction[step])]
common_max=max([8.]+[float(np.max(np.hypot(f[0,z],f[1,z])[mask])) for _,f in fields if mask.any()])
for col,(title,field) in zip(cols,fields):
    fig,ax=plt.subplots(figsize=(5.5,4));speed=np.where(mask,np.hypot(field[0,z],field[1,z]),np.nan);im=ax.imshow(speed,origin='lower',extent=(-3300,3300,-3300,3300),cmap='YlGnBu',vmin=0,vmax=common_max);ax.set(title=title,xlabel='East x (m)',ylabel='North y (m)');fig.colorbar(im,ax=ax,label='m/s');col.pyplot(fig);plt.close(fig)
iy=st.sidebar.slider('Grid y',0,32,16);ix=st.sidebar.slider('Grid x',0,32,16)
fig,ax=plt.subplots(figsize=(10,3));speed=np.where(mask[iy,ix],np.hypot(prediction[:,0,z,iy,ix],prediction[:,1,z,iy,ix]),np.nan);ax.plot(data['nominal_lead_seconds']/60,speed,label='LanWind')
if 'future_uv_ms' in data:
    true=data['future_uv_ms'];vals=np.hypot(true[:,0,z,iy,ix],true[:,1,z,iy,ix]);actual=(data['future_times_utc']-int(data['origin_utc']))/60;ax.plot(actual,vals,label='Observed (actual scan time)')
ax.set(xlabel='Forecast lead (min)',ylabel='Speed (m/s)');ax.legend();st.pyplot(fig);plt.close(fig)
threshold=float(data.get('risk_threshold_s_1',np.nan))
if np.isfinite(threshold) and mask.any():
    shear=horizontal_shear(prediction[step],data['baseline_valid'],float(data.get('grid_spacing_m',cfg['grid']['spacing_m'])))[z];fig,ax=plt.subplots(figsize=(7,3));im=ax.imshow(shear,origin='lower',extent=(-3300,3300,-3300,3300),cmap='magma')
    if np.any(shear>threshold) and np.any(np.isfinite(shear)&(shear<=threshold)):ax.contour(shear>threshold,levels=[.5],colors='cyan',origin='lower',extent=(-3300,3300,-3300,3300))
    fig.colorbar(im,ax=ax,label='s^-1');st.pyplot(fig);plt.close(fig);st.caption('Training-derived visualization threshold. Not an operational warning standard.')
st.caption('300 m is a structural model channel; the published observed-target evaluation covers 100–250 m.')
