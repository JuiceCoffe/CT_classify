import argparse,json,time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,confusion_matrix
from nodule_lab.engine import seed_everything,resolve_device,write_json

class VolDataset(Dataset):
 def __init__(self,path,split,mode):
  with np.load(path,allow_pickle=False) as d: self.x=d[f'{split}_images']; self.y=d[f'{split}_labels'][:,0].astype(np.float32)
  self.mode=mode
 def __len__(self): return len(self.y)
 def __getitem__(self,i):
  v=self.x[i].astype(np.float32)/255.; c=14
  if self.mode=='adjacent': a=np.stack([v[c-1],v[c],v[c+1]])
  elif self.mode=='orthogonal': a=np.stack([v[c],v[:,c,:],v[:,:,c]])
  else: a=v[c][None]
  return torch.from_numpy(a.copy()),torch.tensor(self.y[i])

class CNN25D(nn.Module):
 def __init__(self,in_ch):
  super().__init__(); self.features=nn.Sequential(nn.Conv2d(in_ch,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU()); self.pool=nn.AdaptiveAvgPool2d(1); self.fc=nn.Linear(64,1)
 def forward(self,x): return self.fc(self.pool(self.features(x)).flatten(1)).squeeze(1)
def calc(y,p):
 q=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel(); return {'AUC':float(roc_auc_score(y,p)),'ACC':float(accuracy_score(y,q)),'Sensitivity':float(recall_score(y,q,zero_division=0)),'Specificity':float(tn/(tn+fp) if tn+fp else 0),'F1':float(f1_score(y,q,zero_division=0)),'TP':int(tp),'FP':int(fp),'TN':int(tn),'FN':int(fn)}
def run(data,out,mode,dev,epochs=40):
 seed_everything(42,2); d=resolve_device(dev); tr=VolDataset(data,'train',mode); va=VolDataset(data,'val',mode); te=VolDataset(data,'test',mode); dl=DataLoader(tr,32,shuffle=True); vl=DataLoader(va,64); tl=DataLoader(te,64); m=CNN25D(3 if mode!='single' else 1).to(d); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4); lossf=nn.BCEWithLogitsLoss(); best=-1; stale=0; t0=time.perf_counter()
 for ep in range(1,epochs+1):
  m.train()
  for x,y in dl: x,y=x.to(d),y.to(d); opt.zero_grad(); loss=lossf(m(x),y); loss.backward(); opt.step()
  m.eval(); ys=[];ps=[]
  with torch.no_grad():
   for x,y in vl: ys.append(y.numpy());ps.append(m(x.to(d)).sigmoid().cpu().numpy())
  auc=roc_auc_score(np.concatenate(ys),np.concatenate(ps))
  if auc>best: best=float(auc);stale=0;torch.save({'state_dict':m.state_dict(),'epoch':ep,'val_auc':float(auc)},out/'best.pt')
  else: stale+=1
  if stale>=8: break
 payload=torch.load(out/'best.pt',map_location=d,weights_only=True);m.load_state_dict(payload['state_dict']);m.eval();ys=[];ps=[]; t1=time.perf_counter()
 with torch.no_grad():
  for x,y in tl: ys.append(y.numpy());ps.append(m(x.to(d)).sigmoid().cpu().numpy())
 result=calc(np.concatenate(ys),np.concatenate(ps)); result.update({'epochs':ep,'parameters':sum(p.numel() for p in m.parameters()),'inference_seconds':time.perf_counter()-t1,'train_seconds':time.perf_counter()-t0})
 write_json(out/'metrics.json',result); write_json(out/'config.json',{'mode':mode,'epochs':ep,'input_channels':3 if mode!='single' else 1}); return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='data/raw/nodulemnist3d.npz');p.add_argument('--out',default='outputs/25d');p.add_argument('--device',default='cuda');a=p.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True); r={};
 for mode in ['single','adjacent','orthogonal']:
  z=o/mode;z.mkdir(exist_ok=True);r[mode]=run(a.data,z,mode,a.device)
 write_json(o/'summary.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
