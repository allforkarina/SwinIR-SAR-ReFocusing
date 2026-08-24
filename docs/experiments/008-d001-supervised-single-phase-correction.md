# E008：D001监督式单样本相位校正预测门禁

> 日期：2026-08-24
>
> 状态：实验通过
>
> 前置实验：[E007逐样本相位Oracle](007-d001-per-patch-phase-oracle.md)

## 实验问题

Image可以在训练阶段作为监督，但验证和推理不能读取Image。E008首先只回答一个最小问题：

> SwinIR能否仅从单个Echo的复数频谱，学习预测使该样本接近E007 unrestricted oracle的
> 单位复数相位校正？

单样本通过只证明实现、表示和优化链路可行，不证明多样本泛化。未通过则不扩大数据量。

## 数据流与标签边界

Echo和Image使用同一个Echo RMS归一化，并分别做正交二维FFT和`fftshift`：

```text
X = fftshift(FFT2(Echo / RMS(Echo)))
Y = fftshift(FFT2(Image / RMS(Echo)))
```

模型输入只有`[real(X), imag(X)]`。训练目标为：

```text
P_target = Y * conj(X) / abs(Y * conj(X))
```

网络输出两个通道并逐频率归一化：

```text
P_pred = normalize([channel_0, channel_1])
abs(P_pred) = 1
prediction = IFFT2(ifftshift(P_pred * X))
```

因此网络只能重排Echo频谱相位，不能修改频谱幅度，也不能通过输出整体趋零降低RMSE。
Image只用于构造训练目标和损失；模型前向、验证图和未来推理接口都只接收Echo频谱。

## 损失

总损失由三部分组成：

1. 交叉谱能量平方根加权的圆周相位损失；
2. IFFT重建结果与Image的复数Charbonnier损失；
3. 重建结果与Image的`log1p`幅度L1损失。

默认权重为`1.0 : 0.25 : 0.25`。相位用`cos/sin`单位向量监督，不直接回归存在
`-pi/pi`跳变的角度。

## 成功门槛

同一样本同时计算Echo identity和不使用目标增益的unrestricted phase oracle。Raw模型必须
连续三次同时满足：

- 加权相位alignment至少0.95；
- complex coherence达到oracle的90%；
- SSIM增益至少达到oracle增益的80%；
- edge correlation增益至少达到oracle增益的75%；
- RMSE比oracle最多高0.08；
- 高频能量ratio位于0.75～1.25。

门槛以Raw模型判定，EMA只记录，不允许EMA滞后阻止已成功的过拟合实验。最终人工图按
Echo、Raw、EMA、Oracle、Image五列显示独立峰值和共享Image峰值两种尺度。

## 决策规则

- 通过并且Raw视觉接近Oracle：进入E009多样本监督相位预测；
- 相位alignment高但图像明显不及Oracle：检查频谱幅度差和损失权重；
- 单样本都不能接近Oracle：停止扩大数据量，修改频域网络表示或模型结构；
- 多样本阶段必须使用空间隔离，最终还需完整scene holdout，不能把同一大图的重叠patch
  当作独立泛化证据。

## 产物

- `checkpoints/best.pt`、`latest.pt`、`final.pt`；
- `metrics.jsonl`和`report.json`；
- `figures/step_*.png`五列人工审查图；
- `predictions/step_*.mat`，包含预测图和预测单位相位校正；
- `resolved_config.json`，明确记录推理时Image不可用。

## 服务器结果与人工审查（2026-08-24）

锚点`patch_row_17500_col_9400_2.mat`在step 2000达到连续三次Raw通过并提前停止：

| 指标 | Echo | Oracle | Raw |
| --- | ---: | ---: | ---: |
| normalized complex RMSE | 1.0013 | 0.4477 | 0.4605 |
| complex coherence | 0.0060 | 0.8003 | 0.7887 |
| log-magnitude SSIM | 0.0504 | 0.3000 | 0.3863 |
| edge correlation | 0.0012 | 0.1978 | 0.2511 |
| high-frequency ratio | 0.9375 | 1.0356 | 0.9953 |
| weighted phase alignment | 0.0008 | 1.0000 | 0.9781 |

Raw达到Oracle coherence的98.55%，RMSE只比Oracle高0.0128；SSIM和edge gain分别达到
Oracle增益的134.59%和127.15%。逐步可视化显示道路、地块边界和散射结构从step 500
开始出现，step 2000的Raw已与Oracle具有一致的聚焦形态。项目组人工审查接受该结果。

EMA因固定`0.999`衰减且训练仅2000步而明显滞后，不作为本实验失败证据。E009将保留
Raw主判据，并对EMA增加早期无偏预热。

本实验只证明网络可以在一个训练样本上记忆监督相位校正。尽管前向接口只读取Echo，
模型参数已经编码该样本的Image信息，因此不能作为未见样本推理证据。下一步进入E009
16样本联合过拟合，检验一个共享模型能否同时学习多组内容条件相位映射。
