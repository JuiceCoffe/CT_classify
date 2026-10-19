"""清晰的训练/验证/测试流程；训练不读取 test，评估不更新模型。"""
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import NoduleDataset
from .augment import Compose
from .model import TinyCNN
from .metrics import classification_metrics


def seed_everything(seed, threads):
    """设定随机种子与线程数；同环境同配置可复现，不承诺跨硬件逐位一致。"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name):
    """解析 cpu/cuda/auto，返回 torch.device；显式 CUDA 不可用时报错。"""
    if name not in ('cpu', 'cuda', 'auto'):
        raise ValueError(f'Unknown device={name}')
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu')
    return torch.device(name)


def write_json(path, value):
    """写入可审查的 JSON；不允许静默保存 NaN。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def evaluate(model, loader, device, progress=True):
    """只做前向推理，返回平均 BCE、指标、顺序标签和概率，不修改权重或 BN。"""
    if len(loader.dataset) == 0:
        raise ValueError('evaluate requires nonempty dataset')
    model.eval()
    labels, probabilities = [], []
    loss_sum = 0.
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Evaluate', leave=False, disable=not progress):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss_sum += criterion(logits, targets).item()
            labels.append(targets.cpu().numpy())
            probabilities.append(logits.sigmoid().cpu().numpy())
    labels, probabilities = np.concatenate(labels), np.concatenate(probabilities)
    return loss_sum / len(labels), classification_metrics(labels, probabilities), labels, probabilities


def train(config, progress=True):
    """依据显式配置训练 train/val，返回最佳 checkpoint；输出目录必须不存在。"""
    for key in ('epochs', 'batch_size', 'patience', 'threads'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} expected positive integer, got {config[key]}')
    if config['lr'] <= 0 or config['weight_decay'] < 0:
        raise ValueError('lr must be >0 and weight_decay must be >=0')
    output = Path(config['output'])
    if output.exists():
        raise FileExistsError(f'Output exists: {output}; choose a new --output to preserve experiments')
    seed_everything(config['seed'], config['threads'])
    device = resolve_device(config['device'])
    train_data = NoduleDataset(config['data'], 'train', Compose(config['augmentation']))
    val_data = NoduleDataset(config['data'], 'val')
    if len(torch.unique(val_data.labels)) != 2:
        raise ValueError('Validation AUC checkpoint selection requires both label classes')
    generator = torch.Generator().manual_seed(config['seed'])
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    model = TinyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'config.json', config)
    write_json(output/'environment.json', {'torch': torch.__version__, 'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'parameters': sum(p.numel() for p in model.parameters())})
    best_auc, stale = -1., 0
    checkpoint = output/'best.pt'
    with (output/'history.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['epoch','train_loss','val_loss','val_auc','val_f1'])
        writer.writeheader()
        for epoch in range(1, config['epochs'] + 1):
            model.train(); total_loss = 0.
            for images, targets in tqdm(train_loader, desc=f'Train {epoch}/{config["epochs"]}', disable=not progress):
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward(); optimizer.step()
                total_loss += loss.item() * len(targets)
            val_loss, metrics, _, _ = evaluate(model, val_loader, device, progress)
            row = dict(epoch=epoch, train_loss=total_loss/len(train_data), val_loss=val_loss,
                       val_auc=metrics['AUC'], val_f1=metrics['F1'])
            writer.writerow(row); file.flush()
            print(f'epoch={epoch} train_loss={row["train_loss"]:.4f} val_loss={val_loss:.4f} val_auc={metrics["AUC"]:.4f}')
            # 严格提高才替换，AUC 相同保留较早权重。测试集不参与此过程。
            if metrics['AUC'] > best_auc:
                best_auc, stale = metrics['AUC'], 0
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'val_auc': best_auc,
                            'config': config, 'model': 'TinyCNN'}, checkpoint)
            else:
                stale += 1
            if stale >= config['patience']:
                break
    write_json(output/'training_complete.json', {'completed_epochs': epoch, 'best_val_auc': best_auc,
                                                'test_accessed': False})
    return checkpoint


def load_model(checkpoint, device):
    """读取本项目 checkpoint，返回 eval 状态的 TinyCNN 与训练元数据。"""
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload['model'] != 'TinyCNN':
        raise ValueError(f'Unsupported checkpoint model={payload["model"]}')
    model = TinyCNN().to(device)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, payload
