# E015：64 样本相位监督目标对照

## 目的

比较当前相位阶段中辅助损失的监督对象，判断完整 Image 的频谱幅度差异是否会干扰相位校正学习。

所有组的模型输入都只有 Echo 的复频谱，主监督标签都相同：

`unit(Image_spectrum * conj(Echo_spectrum))`

phase-only Oracle 只是训练标签或辅助损失目标，从不作为推理输入。最终评估始终以完整 Image 为参考，并以 phase-only Oracle 作为仅相位校正的上限基线。

## 三组对照

| 组别 | 辅助复数/幅度目标 | loss 权重 |
|---|---|---|
| A | 完整 Image | phase=1，complex=0.25，log-magnitude=0.25 |
| B | phase-only Oracle | phase=1，complex=0.25，log-magnitude=0.25 |
| C | 不使用辅助损失 | phase=1，complex=0，log-magnitude=0 |

## 受控条件

- 新数据集上确定性、空间分散且互不重叠的同一组 64 个 patch；
- 每个 epoch 对 64 个 patch 各更新一次，不按亮度或能量改变样本权重；
- 三组均从相同 seed=42 的随机初始化开始；
- 每组 19,200 optimizer step，即每个样本平均更新 300 次；
- step 0、3200、6400、9600、12800、16000、19200 评估并保存可视化；
- 不因达到阈值提前终止。

E015 是训练/拟合效率诊断，不包含未见样本，不能用于声称空间泛化或跨场景泛化。泛化实验须在后续完成空间隔离的数据划分后单独进行。
