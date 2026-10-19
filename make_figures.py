import json,csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,confusion_matrix
from nodule_lab.data import NoduleDataset
def readpred(p):
 with open(p) as f: rows=list(csv.DictReader(f))
 return np.array([int(r['label']) for r in rows]),np.array([float(r['probability']) for r in rows])
root=Path('outputs'); out=root/'figures'; out.mkdir(parents=True,exist_ok=True)
data='data/raw/nodulemnist3d.npz'; ds=[NoduleDataset(data,s) for s in ('train','val','test')]
fig,ax=plt.subplots(1,3,figsize=(9,3));
for a,d,s in zip(ax,ds,('Train','Val','Test')):
 c=np.bincount(d.labels.numpy().astype(int),minlength=2); a.bar(['Class 0','Class 1'],c,color=['#4472c4','#ed7d31']); a.set_title(f'{s} n={len(d)}'); a.set_ylabel('Count')
fig.tight_layout(); fig.savefig(out/'class_distribution.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(2,3,figsize=(9,6)); x=ds[0].images[0][0].numpy(); vol=np.load(data)['train_images'][0]/255.
for a,im,t in zip(ax[0],[vol[14],vol[:,14,:],vol[:,:,14]],['Axial','Coronal','Sagittal']): a.imshow(im,cmap='gray'); a.set_title(t); a.axis('off')
ax[1,0].hist(vol.ravel(),bins=32,color='#4472c4'); ax[1,0].set_title('Gray histogram'); ax[1,0].set_xlabel('Normalized intensity');
for a in ax[1,1:]: a.axis('off')
fig.tight_layout(); fig.savefig(out/'typical_views.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots();
for name in ['baseline_test','augment_test']:
 y,p=readpred(root/name/'predictions.csv'); fpr,tpr,_=roc_curve(y,p); auc=float(json.loads((root/name/'metrics.json').read_text())['AUC']); ax.plot(fpr,tpr,label=f'{name} AUC={auc:.3f}')
ax.plot([0,1],[0,1],'k--'); ax.set(xlabel='False positive rate',ylabel='True positive rate'); ax.legend(); fig.tight_layout(); fig.savefig(out/'roc_baseline_augment.png',dpi=180); plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8,3));
for a,name in zip(axs,['baseline_test','augment_test']):
 y,p=readpred(root/name/'predictions.csv'); cm=confusion_matrix(y,p>=.5); a.imshow(cm,cmap='Blues'); a.set_title(name); a.set_xlabel('Predicted'); a.set_ylabel('True');
 for (i,j),v in np.ndenumerate(cm): a.text(j,i,str(v),ha='center',va='center')
fig.tight_layout(); fig.savefig(out/'confusion_baseline_augment.png',dpi=180); plt.close(fig)
