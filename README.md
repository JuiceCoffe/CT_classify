# NoduleMNIST3D 学生基础代码

本项目提供**能独立运行的二维分类基础流程**：官方数据读取、三种可扩展训练增强、TinyCNN、训练、验证与测试。它不是全部作业答案：传统特征分类、不平衡处理、Grad-CAM、合成、半监督和2.5D由学生按任务要求探索。发布包不含历史成绩、权重或参考预测。

## 文件结构

```text
student_code/
  prepare_data.py       # 官方数据校验、可选下载和统计
  train.py              # 训练入口，只读取 train/val
  evaluate.py           # 独立验证或测试入口
  configs/
    baseline.json      # 不增强
    augment.json       # 三种增强组合示例，不保证优于基线
  nodule_lab/
    data.py            # NPZ读取、中心切片、归一化
    augment.py         # 翻转、噪声、对比度与扩展登记表
    model.py           # TinyCNN，输出logits
    metrics.py         # 固定阈值指标及混淆计数
    engine.py          # 训练循环、验证、checkpoint
  tests/test_basics.py  # 数据、增强、模型、评估和训练流程测试
  requirements.txt
```

所有命令均在 `student_code/` 下执行。已有环境包含依赖时直接使用；无需为本项目另建环境。基本依赖为 Python 3.10+、PyTorch 2.4+、NumPy、scikit-learn 和 tqdm；无需 torchvision。

```bash
python -m pip install -r requirements.txt  # 仅在环境缺少依赖时使用
```

## 1. 准备官方数据

将 `nodulemnist3d.npz` 放入 `data/raw/`。也可以显式下载：

```bash
python prepare_data.py --download
```

已经有文件时，只校验与统计，不联网：

```bash
python prepare_data.py --data /home/user/datasets/nodulemnist3d.npz
```

官方文件地址：<https://zenodo.org/records/10519652/files/nodulemnist3d.npz?download=1>。
MD5 为 `8755a7e9e05a4d9ce80a24c3e7a256f3`。网络不可用时手动下载再校验；代码不以随机数据冒充数据集。

保留官方 train/val/test。单个体积 `[28,28,28]` 取第14个索引中心切片，`uint8` 转 `float32` 后除以255，返回 `[1,28,28]` 和标量标签；batch为 `[B,1,28,28]`、`[B]`。标签1为高恶性可疑度，不等于病理确诊。读取样本前的统计不使用测试成绩来调参。

## 2. 先跑一个短程检查

```bash
python train.py --config configs/baseline.json --device cpu --epochs 2 --output outputs/check_baseline
python evaluate.py --checkpoint outputs/check_baseline/best.pt --split val --device cpu --output outputs/check_val
```

外部数据文件需要在每条命令中显式增加 `--data /home/user/datasets/nodulemnist3d.npz`。短程运行只用于验证代码，不代表完成正式实验。所有训练/评估输出目录必须是新目录，不会自动覆盖或断点续训；重跑请换目录。

## 3. 正式训练与最终测试

```bash
python train.py --config configs/baseline.json --device cpu
python evaluate.py --checkpoint outputs/baseline/best.pt --split test --device cpu --output outputs/baseline_test
```

有可用GPU时，可将 `cpu` 改为 `cuda`；`auto` 自动选择设备。基础代码实际验收范围见 `验证说明.txt`，不把CPU验证当作GPU验证。

固定阈值0.5，概率 `>=0.5` 判为类别1，不做阈值优化。模型返回原始logits，训练使用 `BCEWithLogitsLoss`，评估时才调用sigmoid。最佳权重仅由验证AUC选择，相同AUC保留较早权重。训练过程不读测试集，只有显式运行 `evaluate.py --split test` 才做测试评价。

默认：seed=42、Adam、lr=0.001、weight_decay=0.0001、batch=32、最多40轮、patience=8、CPU线程2、DataLoader workers=0。JSON保存所有实验参数；命令行只覆盖明确传入的选项。`data`/`output` 配置中的相对路径以本项目根目录为基准；评估入口的显式相对路径以当前目录为基准，推荐始终在项目目录运行。

训练输出：

- `best.pt`：模型权重与最佳epoch、验证AUC、实际配置；用于推理，不是优化器续训快照。
- `history.csv`：每轮训练损失、验证损失、验证AUC和F1，可由学生绘图。
- `config.json`、`environment.json`：实际参数与设备。
- `training_complete.json`：完成轮数和训练未访问测试集的标记。

评估输出：`metrics.json`（指标、BCE、固定阈值）、`predictions.csv`（按原始样本顺序）、`evaluation.json`（检查点与数据来源）。单类子集的AUC为null，其他分母为零的指标记0；正式验证集必须包含两类以支持AUC选模。

## 4. 三种训练增强

| 配置名称 | 方法 | 默认参数 |
|---|---|---|
| `flip` | 左右与上下各自独立随机翻转 | 每个方向概率0.5 |
| `noise` | 加零均值高斯噪声，裁剪至[0,1] | 概率0.5，sigma在[0.01,0.08]随机取值 |
| `contrast` | 围绕单张图的均值调整对比度，裁剪至[0,1] | 概率0.5，系数在[0.65,1.15]随机取值 |

```bash
python train.py --config configs/augment.json --device cpu
python evaluate.py --checkpoint outputs/augmented/best.pt --split test --device cpu --output outputs/augmented_test
```

`augmentation: []` 表示不增强。组合配置按列表顺序执行，**可在同一图上叠加多个增强**，不是三选一。这是当前数据增强探索的示例，与旧退化测试实验不同；验证、测试保持原始图像，Dataset会拒绝给它们传入transform。

单独比较某一种增强时，复制JSON到新文件，仅保留相应配置项，同时换一个输出目录。例如：

```json
"augmentation": [{"name": "noise", "p": 0.5, "sigma": [0.01, 0.08]}]
```

以上只是完整JSON中的一个字段。其余训练设置保持一致，解释增强的医学合理性，不承诺任何增强一定提高成绩。

扩展新增强时：

1. 在 `augment.py` 增加一个类，实现 `__call__(image)`；输入输出都为float32 `[1,28,28]`、有限值且在[0,1]。
2. 在 `TRANSFORMS` 登记名称，JSON列表中加入该名称和参数。
3. 使用torch随机数，保证 `torch.manual_seed` 能控制随机性；不修改标签。
4. 在测试中验证形状、范围、概率边界与复现性。未知方法名或拼错参数会报错。

## 5. 运行测试

完全离线测试（使用明确标记的合成夹具，不需要下载数据）：

```bash
python -m unittest discover -s tests -v
```

加入官方文件的集成检查：

```bash
NODULE_DATA=/home/user/datasets/nodulemnist3d.npz python -m unittest discover -s tests -v
```

覆盖内容：中心片/归一化、划分与标签校验、禁止验证测试增强、原图不被增强污染；三种增强的数学行为/边界/复现；固定阈值混淆矩阵；模型形状/反向梯度/参数更新；验证不改权重和BN；checkpoint回放；AUC并列及早停；同种子复现；命令行覆盖和输出保护；独立评估CLI逐例导出。设置 `NODULE_DATA` 后另检查官方MD5、类别数和中心切片。

测试目录与程序均使用临时输出，不删除或覆盖已有实验。真实数据短程训练检查由教师另行执行；学生仍需自行完成正式任务。当前只提供基础2D流程，未实现的作业方向不以空函数或伪结果代替。
