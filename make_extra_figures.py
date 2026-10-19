from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from nodule_lab.data import NoduleDataset
root=Path('outputs'); out=root/'figures'; out.mkdir(exist_ok=True)
rows=[]
with (root/'baseline_gpu'/'history.csv').open() as f: rows=list(csv.DictReader(f))
ep=[int(r['epoch']) for r in rows]; tr=[float(r['train_loss']) for r in rows]; vl=[float(r['val_loss']) for r in rows]; va=[float(r['val_auc']) for r in rows]
fig,ax=plt.subplots(1,2,figsize=(9,3.5)); ax[0].plot(ep,tr,label='Train loss'); ax[0].plot(ep,vl,label='Val loss'); ax[0].set(xlabel='Epoch',ylabel='BCE loss'); ax[0].legend(); ax[1].plot(ep,va,color='#c55a11'); ax[1].set(xlabel='Epoch',ylabel='Val AUC'); fig.tight_layout(); fig.savefig(out/'training_curves.png',dpi=180); plt.close(fig)
with np.load('data/raw/nodulemnist3d.npz',allow_pickle=False) as d: v=d['train_images'][0].astype(np.float32)/255.; c=14
fig,ax=plt.subplots(1,3,figsize=(9,3)); ims=[np.stack([v[c-1],v[c],v[c+1]]),np.stack([v[c],v[:,c,:],v[:,:,c]]),v[c][None]]; titles=['相邻切片三通道','正交切面三通道','单张中心切片']
for a,im,t in zip(ax,ims,titles): a.imshow(np.transpose(im,(1,2,0)) if im.shape[0]==3 else im[0],cmap='gray'); a.set_title(t); a.axis('off')
fig.tight_layout(); fig.savefig(out/'25d_input_examples.png',dpi=180); plt.close(fig)
