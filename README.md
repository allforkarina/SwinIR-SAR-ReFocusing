# SwinIR-SAR ReFocusing

本项目按照
[实施规范](docs/specifications/SwinIR-SAR_重聚焦实施规范_v0.1.md)与
[实施计划](docs/specifications/SwinIR-SAR_重聚焦实施计划_v0.1.md)
独立实现 SwinIR 网络架构，用于后续 SAR 实部/虚部双通道同尺寸重聚焦实验。

当前实现包含模型、严格配对 Dataset 与训练入口；尚不包含独立的 `test.py` 推理/评价脚本。

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 激活命令为：

```powershell
.\.venv\Scripts\Activate.ps1
```

## 快速检查

```bash
pytest
python scripts/inspect_model.py
```

## SAR 训练

训练配置位于 `configs/train_sar.yaml`。在服务器上先将其中的
`data.echo_dir` 和 `data.image_dir` 改为真实的 MAT 数据目录，再安装依赖：

```bash
python -m pip install -r requirements.txt
python main.py --config configs/train_sar.yaml --run-name sar_baseline_v1
```

该入口会自动建立基于坐标的训练/保护带/验证集清单，使用每个 echo patch 自身
的 RMS 同时归一化 echo 与对应 image，并把运行产物写入
`runs/<run-name>/`。`guard` 样本只用于隔离，不参与训练或验证。

恢复必须显式指定 checkpoint：

```bash
python main.py --config configs/train_sar.yaml --run-name sar_baseline_v1 \
  --resume runs/sar_baseline_v1/checkpoints/latest.pt
```

恢复时会严格比对完整配置与数据 manifest 指纹；任一不一致都会失败，而不会混合
两次不同的数据划分或训练设定。

## 使用

```python
import torch
import yaml

from swinir import SwinIR

with open("configs/swinir_same_size.yaml", encoding="utf-8") as file:
    config = yaml.safe_load(file)["model"]

model = SwinIR(**config)
x = torch.randn(1, 2, 64, 64)
y = model(x)
assert y.shape == x.shape
```

标准配置为 6 个 RSTB、每个 RSTB 6 个交替 W-MSA/SW-MSA Block，
输入输出均为 `[B, 2, H, W]`。模型会将动态输入反射填充到窗口大小的整数倍，
推理完成后裁剪回原始尺寸；极小输入会安全地改用复制填充。

## 官方对照

运行时不依赖官方 SwinIR。若要执行结构、参数量和 FP32 数值等价测试，
请按 `references/README.md` 放置固定版本的官方参考文件，然后运行：

```bash
python scripts/compare_official.py
```
