"""训练入口：只使用 train 和 val，不在训练结束时自动访问测试集。"""
import argparse
import json
from pathlib import Path
from nodule_lab.engine import train

ROOT = Path(__file__).resolve().parent


def read_config(args):
    """读取 JSON 并应用显式 CLI 覆盖；相对数据/输出路径以项目根目录为基准。"""
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    required = {'data','output','device','seed','epochs','batch_size','lr','weight_decay','patience','threads','augmentation'}
    if set(config) != required:
        raise ValueError(f'Config keys mismatch: missing={required-set(config)}, unknown={set(config)-required}')
    for key in ['data','output','device','epochs','seed']:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for key in ['data','output']:
        path = Path(config[key])
        config[key] = str((ROOT/path).resolve() if not path.is_absolute() else path)
    return config


def main():
    """解析参数并启动训练；使用 --help 查看可覆盖配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    parser.add_argument('--data')
    parser.add_argument('--output')
    parser.add_argument('--device', choices=['cpu','cuda','auto'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    print('Best checkpoint:', train(read_config(args)))


if __name__ == '__main__':
    main()
