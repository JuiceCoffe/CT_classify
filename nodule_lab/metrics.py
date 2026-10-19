"""二分类指标；统一使用概率 >= 0.5 的固定分类规则。"""
import numpy as np
from sklearn.metrics import roc_auc_score


def classification_metrics(labels, probabilities):
    """输入两个非空 [N] 数组，返回指标和混淆计数；单类时 AUC 为 None。"""
    y, p = np.asarray(labels), np.asarray(probabilities)
    if y.ndim != 1 or y.shape != p.shape or len(y) == 0:
        raise ValueError(f'metrics expected matching nonempty [N], got labels={y.shape}, probabilities={p.shape}')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('metrics expected binary labels and finite probabilities in [0,1]')
    pred = p >= .5
    tp = int(((y == 1) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.
    specificity = tn / (tn + fp) if tn + fp else 0.
    return {'AUC': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'ACC': (tp + tn) / len(y), 'Precision': tp / (tp + fp) if tp + fp else 0.,
            'Sensitivity': sensitivity, 'Specificity': specificity,
            'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'BACC': (sensitivity + specificity) / 2, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
