import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, confusion_matrix
from nodule_lab.data import NoduleDataset
from nodule_lab.model import TinyCNN
from nodule_lab.engine import seed_everything, resolve_device, evaluate, write_json
from nodule_lab.augment import Compose

def metrics(y, p):
    pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'AUC':float(roc_auc_score(y,p)),'ACC':float(accuracy_score(y,pred)),
            'Sensitivity':float(recall_score(y,pred,zero_division=0)),
            'Specificity':float(tn/(tn+fp) if tn+fp else 0),'F1':float(f1_score(y,pred,zero_division=0)),
            'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}

def traditional(data, out):
    def feats(ds):
        x=ds.images.numpy()[:,0]
        h=np.stack([np.histogram(a,bins=16,range=(0,1),density=True)[0] for a in x])
        stats=np.stack([x.mean((1,2)),x.std((1,2)),x.min((1,2)),x.max((1,2)),
                        np.quantile(x,.25,axis=(1,2)),np.quantile(x,.5,axis=(1,2)),np.quantile(x,.75,axis=(1,2))],1)
        # downsampled intensity pattern retains shape while keeping the classical model small
        small=x[:,::2,::2].reshape(len(x),-1)
        return np.concatenate([h,stats,small],1)
    tr,va,te=[NoduleDataset(data,s) for s in ('train','val','test')]
    Xtr,Xv,Xte=map(feats,(tr,va,te)); ytr,yv,yt=[d.labels.numpy().astype(int) for d in (tr,va,te)]
    pipe=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42))]); pipe.fit(Xtr,ytr)
    pv=pipe.predict_proba(Xv)[:,1]; pt=pipe.predict_proba(Xte)[:,1]
    write_json(out/'traditional_val.json',metrics(yv,pv)); write_json(out/'traditional_test.json',metrics(yt,pt))
    return metrics(yt,pt)

def train_custom(data, out, mode, device, epochs=40):
    seed_everything(42,2); dev=resolve_device(device)
    tr=NoduleDataset(data,'train',Compose([])); va=NoduleDataset(data,'val'); te=NoduleDataset(data,'test')
    if mode=='weighted_bce':
        counts=np.bincount(tr.labels.numpy().astype(int),minlength=2); pos_weight=torch.tensor([counts[0]/counts[1]],device=dev)
        sampler=None; criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif mode=='oversample':
        counts=np.bincount(tr.labels.numpy().astype(int),minlength=2); weights=np.array([1/counts[int(y)] for y in tr.labels.numpy()]); sampler=WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),len(tr),replacement=True); criterion=nn.BCEWithLogitsLoss()
    else: sampler=None; criterion=nn.BCEWithLogitsLoss()
    loader=DataLoader(tr,batch_size=32,shuffle=sampler is None,sampler=sampler,num_workers=0); vl=DataLoader(va,32); tl=DataLoader(te,32)
    m=TinyCNN().to(dev); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4); best=-1; stale=0
    out.mkdir(parents=True,exist_ok=True)
    for ep in range(1,epochs+1):
        m.train()
        for x,y in loader:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); loss=criterion(m(x),y); loss.backward(); opt.step()
        _,mv,_,_=evaluate(m,vl,dev,progress=False)
        if mv['AUC']>best: best=mv['AUC']; stale=0; torch.save({'state_dict':m.state_dict(),'epoch':ep,'val_auc':best,'model':'TinyCNN'},out/'best.pt')
        else: stale+=1
        if stale>=8: break
    _,mt,labels,probs=evaluate(m,tl,dev,progress=False)
    # evaluate selected checkpoint for consistency
    payload=torch.load(out/'best.pt',map_location=dev,weights_only=True); m.load_state_dict(payload['state_dict']); _,mt,labels,probs=evaluate(m,tl,dev,progress=False)
    write_json(out/'metrics.json',mt); write_json(out/'config.json',{'mode':mode,'epochs':ep,'pos_weight':float(pos_weight.item()) if mode=='weighted_bce' else None})
    return mt

def gradcam(data, ckpt, out, device):
    import matplotlib.pyplot as plt
    dev=resolve_device(device); ds=NoduleDataset(data,'test'); loader=DataLoader(ds,32,shuffle=False)
    m=TinyCNN().to(dev); m.load_state_dict(torch.load(ckpt,map_location=dev,weights_only=True)['state_dict']); m.eval(); acts=[]; grads=[]
    def fh(_,__,o): acts.append(o)
    def bh(_,gin,gout): grads.append(gout[0])
    h1=m.features[-2].register_forward_hook(fh); h2=m.features[-2].register_full_backward_hook(bh)
    xs=[]; ys=[]; probs=[]; cams=[]
    for x,y in loader:
        x=x.to(dev); m.zero_grad(); z=m(x); p=z.sigmoid(); z.sum().backward();
        a=acts[-1].detach(); g=grads[-1].detach(); w=g.mean((2,3),keepdim=True); cam=(w*a).sum(1).relu(); cam=cam/(cam.amax((1,2),keepdim=True)+1e-8)
        xs.append(x.detach().cpu()); ys.append(y); probs.append(p.detach().cpu()); cams.append(cam.cpu())
    x=torch.cat(xs); y=torch.cat(ys).numpy().astype(int); pp=torch.cat(probs).numpy(); cc=torch.cat(cams).numpy(); pred=(pp>=.5).astype(int)
    cm=confusion_matrix(y,pred,labels=[0,1]); cases={}
    for kind,mask in [('TP',(y==1)&(pred==1)),('TN',(y==0)&(pred==0)),('FP',(y==0)&(pred==1)),('FN',(y==1)&(pred==0))]:
        idx=np.where(mask)[0]
        if len(idx):
            i=int(idx[0]); cases[kind]={'index':i,'label':int(y[i]),'probability':float(pp[i])}
            fig,ax=plt.subplots(1,3,figsize=(9,3)); im=x[i,0].numpy(); heat=cc[i];
            ax[0].imshow(im,cmap='gray'); ax[1].imshow(heat,cmap='jet'); ax[2].imshow(im,cmap='gray'); ax[2].imshow(heat,cmap='jet',alpha=.45)
            for a,t in zip(ax,['Original','Grad-CAM','Overlay']): a.set_title(t); a.axis('off')
            fig.suptitle(f'{kind} label={y[i]} p={pp[i]:.3f}'); fig.tight_layout(); fig.savefig(out/f'gradcam_{kind}.png',dpi=160); plt.close(fig)
    write_json(out/'gradcam_cases.json',cases); h1.remove(); h2.remove()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/raw/nodulemnist3d.npz'); ap.add_argument('--out',default='outputs/required'); ap.add_argument('--baseline-ckpt',required=True); ap.add_argument('--device',default='cuda'); args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    results={'traditional_test':traditional(args.data,out)}
    for mode in ('weighted_bce','oversample'): results[mode]=train_custom(args.data,out/mode,mode,args.device)
    gradcam(args.data,args.baseline_ckpt,out,args.device); write_json(out/'summary.json',results); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
