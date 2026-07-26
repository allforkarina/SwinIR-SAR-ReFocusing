# SwinIR-SAR ReFocusing

本项目按照
[实施规范](docs/specifications/SwinIR-SAR_重聚焦实施规范_v0.1.md)与
[实施计划](docs/specifications/SwinIR-SAR_重聚焦实施计划_v0.1.md)
独立实现 SwinIR 网络架构，用于后续 SAR 实部/虚部双通道同尺寸重聚焦实验。

当前阶段仅包含模型架构与验证，不包含数据集、损失函数、优化器或正式训练。

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
