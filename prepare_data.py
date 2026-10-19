"""校验/下载官方数据；保存统计，不训练模型。默认不联网，下载需显式 --download。"""
import argparse
import hashlib
import urllib.request
from pathlib import Path
import numpy as np
from tqdm import tqdm
from nodule_lab.data import MD5, URL, NoduleDataset
from nodule_lab.engine import write_json

ROOT = Path(__file__).resolve().parent


def check_md5(path):
    """返回官方文件摘要；不匹配时给出实际 MD5 并报错。"""
    digest = hashlib.md5()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != MD5:
        raise ValueError(f'{path}: MD5 expected={MD5}, actual={digest.hexdigest()}')
    return digest.hexdigest()


def main():
    """校验官方体积及划分数量，输出数据摘要 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/raw/nodulemnist3d.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/data_summary.json')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if not args.data.exists():
        if not args.download:
            raise FileNotFoundError(f'{args.data}; place official NPZ there or add --download. URL={URL}')
        args.data.parent.mkdir(parents=True, exist_ok=True)
        partial = args.data.with_suffix('.part')
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open('wb') as file:
            with tqdm(total=int(response.headers.get('Content-Length',0)) or None, unit='B', unit_scale=True, desc='Download') as bar:
                while True:
                    block = response.read(1024*1024)
                    if not block: break
                    file.write(block); bar.update(len(block))
        check_md5(partial)
        partial.rename(args.data)
    digest = check_md5(args.data)
    summary = []
    for split, expected in tqdm([('train',1158),('val',165),('test',310)], desc='Check splits'):
        dataset = NoduleDataset(args.data, split)
        if len(dataset) != expected:
            raise ValueError(f'{split} expected {expected} samples, got {len(dataset)}')
        counts = np.bincount(dataset.labels.numpy().astype(int), minlength=2)
        summary.append({'split': split, 'count': len(dataset), 'class_counts': counts.tolist(),
                        'input_shape': list(dataset.images.shape), 'range': [float(dataset.images.min()),float(dataset.images.max())]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {'md5': digest, 'splits': summary})
    print(summary)


if __name__ == '__main__':
    main()
