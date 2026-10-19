"""独立评估入口：加载最佳权重，在 val 或 test 上输出指标与逐例预测。"""
import argparse
import csv
from pathlib import Path
from torch.utils.data import DataLoader
from nodule_lab.data import NoduleDataset
from nodule_lab.engine import resolve_device, load_model, evaluate, write_json, seed_everything

ROOT = Path(__file__).resolve().parent


def main():
    """显式选择 checkpoint 和 split；预测不增强，不覆盖已有评估目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT/'data/raw/nodulemnist3d.npz')
    parser.add_argument('--split', choices=['val','test'], default='test')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda','auto'], default='auto')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Output exists: {args.output}; choose a new --output')
    seed_everything(42, 2)
    device = resolve_device(args.device)
    model, payload = load_model(args.checkpoint, device)
    data = NoduleDataset(args.data, args.split)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    loss, metrics, labels, probabilities = evaluate(model, loader, device)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'metrics.json', {'split': args.split, 'threshold': .5, 'loss': loss, **metrics})
    write_json(args.output/'evaluation.json', {'checkpoint': str(args.checkpoint.resolve()),
        'selected_epoch': payload['epoch'], 'data': str(args.data.resolve()), 'augmentation': False})
    with (args.output/'predictions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['index','label','probability','prediction'])
        for index, (label, probability) in enumerate(zip(labels, probabilities)):
            writer.writerow([index, int(label), float(probability), int(probability >= .5)])
    print(metrics)


if __name__ == '__main__':
    main()
