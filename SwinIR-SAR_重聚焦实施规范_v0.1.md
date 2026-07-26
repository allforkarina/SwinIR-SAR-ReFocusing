# SwinIR-SAR 重聚焦实施规范（Implementation Spec）

> 版本：v0.1  
> 状态：第一阶段基线实施规范  
> 目标：忠实复现 SwinIR 主体结构，并验证其对 PFA 散焦 SAR 图像的同尺寸重聚焦能力  
> 原则：模型中间主干遵循原论文设计；SAR 输入输出形式暂未最终确定，第一版暂以实部/虚部双通道作为预设方案

---

## 1. 项目背景

在近场、复杂运动轨迹或平面波假设不成立的情况下，PFA 使用线性空间频率映射进行成像，会因高阶相位误差、空间变化误差和残余距离徙动产生目标散焦。

本项目不在第一阶段显式求解这些复杂误差，而是验证以下映射是否可以通过 SwinIR 学习：

\[
X_{\mathrm{PFA}} \longrightarrow X_{\mathrm{focus}}
\]

其中：

- \(X_{\mathrm{PFA}}\)：由 PFA 得到的散焦 SAR 图像；
- \(X_{\mathrm{focus}}\)：由精确成像算法得到的聚焦参考图像；
- 输入和输出在第一阶段保持相同空间尺寸；
- 任务定义为 **SAR 图像重聚焦（refocusing）**，而不是传统意义上的图像放大超分辨率。

核心假设为：

\[
\hat X_{\mathrm{focus}}
=
X_{\mathrm{PFA}}
+
N_\theta(X_{\mathrm{PFA}})
\]

网络主要学习散焦图像相对于聚焦图像的修正量。

---

## 2. 第一阶段研究问题

第一阶段只回答以下问题：

> 在训练集与测试集成像参数分布基本一致、输入和标签严格配准、复数信息得到保留的条件下，原始 SwinIR 主干是否能够学习 PFA 散焦图像到聚焦图像的映射？

第一阶段暂不回答：

- 模型能否泛化到完全不同的轨迹；
- 模型能否泛化到不同载频、带宽、孔径和距离；
- 模型能否适配实测数据；
- 模型是否满足严格的回波数据一致性；
- 模型能否突破雷达原始带宽与孔径决定的物理分辨率。

---

## 3. 总体实施原则

### 3.1 原论文主体保持不变

以下模块应尽可能忠实复现原论文设计：

- 浅层 \(3\times3\) 卷积；
- Residual Swin Transformer Block（RSTB）；
- Swin Transformer Layer（STL）；
- Window Multi-head Self-Attention（W-MSA）；
- Shifted Window Multi-head Self-Attention（SW-MSA）；
- 相对位置偏置；
- MLP；
- LayerNorm；
- STL 内部残差连接；
- RSTB 内部残差连接；
- 深层主干长残差连接；
- RSTB 后的 \(3\times3\) 卷积；
- 多个 RSTB 后的 \(3\times3\) 卷积；
- 同尺寸图像恢复重建头；
- Charbonnier 损失作为第一版默认损失。

### 3.2 只进行 SAR 任务所必需的适配

第一版允许修改：

- 输入通道数；
- 输出通道数；
- 数据读取与预处理；
- SAR 输入、标签生成方式；
- 相位与幅度对齐；
- SAR 评价指标。

中间特征维度、RSTB 结构、STL 结构和窗口注意力机制不因 SAR 任务而改动。

---

## 4. 任务定义

### 4.1 输入

输入为 PFA 形成的散焦 SAR 图像。

输入形式目前尚未最终确定，第一版暂定为：

\[
X_{\mathrm{in}}
=
\begin{bmatrix}
\operatorname{Re}(X_{\mathrm{PFA}}) \\
\operatorname{Im}(X_{\mathrm{PFA}})
\end{bmatrix}
\in \mathbb{R}^{2\times H\times W}
\]

即：

- 通道 0：实部；
- 通道 1：虚部。

该方案只是当前预设，不作为最终固定设计。

后续可对照的候选输入形式包括：

1. 实部、虚部；
2. 幅度、相位；
3. 对数幅度、相位；
4. 幅度、\(\cos\phi\)、\(\sin\phi\)；
5. 实部、虚部、幅度、相位等多通道组合。

### 4.2 输出

输出形式同样暂未最终确定。

第一版预设输出为聚焦复图像的实部与虚部：

\[
\hat X_{\mathrm{out}}
=
\begin{bmatrix}
\operatorname{Re}(\hat X_{\mathrm{focus}}) \\
\operatorname{Im}(\hat X_{\mathrm{focus}})
\end{bmatrix}
\in \mathbb{R}^{2\times H\times W}
\]

对应复数结果：

\[
\hat X_{\mathrm{focus}}
=
\hat X_R+j\hat X_I
\]

第一阶段保持：

\[
H_{\mathrm{out}}=H_{\mathrm{in}},
\qquad
W_{\mathrm{out}}=W_{\mathrm{in}}
\]

不进行空间尺寸放大。

### 4.3 残差输出形式

默认采用图像级残差学习：

\[
R=N_\theta(X_{\mathrm{in}})
\]

\[
\hat X_{\mathrm{out}}=X_{\mathrm{in}}+R
\]

其中：

- \(X_{\mathrm{in}}\) 是双通道实值表示；
- \(R\) 是网络预测的双通道修正量；
- 相加发生在同一表示空间中。

由于输入输出形式尚未最终确定，该残差接口应设计为可配置项。

---

## 5. 模型结构规范

### 5.1 浅层特征提取

输入经过一个 \(3\times3\) 卷积：

\[
F_0=H_{\mathrm{SF}}(X_{\mathrm{in}})
\]

默认配置：

| 参数 | 默认值 |
|---|---:|
| 输入通道数 | 2（暂定） |
| 输出特征通道数 | 180 |
| 卷积核 | \(3\times3\) |
| 步长 | 1 |
| padding | 1 |
| 是否改变空间尺寸 | 否 |

要求：

- 输入通道数必须通过配置文件指定；
- 不允许在模型代码中写死为 2 或 3；
- 中间 embedding dimension 默认保持原论文标准配置。

### 5.2 深层特征提取主干

默认采用：

| 参数 | 默认值 |
|---|---:|
| RSTB 数量 | 6 |
| 每个 RSTB 中 STL 数量 | 6 |
| 窗口大小 | 8 |
| shift size | 4 |
| embedding dimension | 180 |
| attention heads | 6 |
| MLP ratio | 2 |
| 激活函数 | GELU |
| 归一化 | LayerNorm |
| RSTB 卷积核 | \(3\times3\) |
| 深层主干末尾卷积核 | \(3\times3\) |

主干计算过程：

\[
F_1=H_{\mathrm{RSTB}_1}(F_0)
\]

\[
F_2=H_{\mathrm{RSTB}_2}(F_1)
\]

\[
\cdots
\]

\[
F_K=H_{\mathrm{RSTB}_K}(F_{K-1})
\]

\[
F_{\mathrm{DF}}=H_{\mathrm{Conv}}(F_K)
\]

\[
F_{\mathrm{fusion}}=F_0+F_{\mathrm{DF}}
\]

### 5.3 STL 结构

每个 STL 必须实现：

\[
X'
=
X+
\operatorname{MSA}(\operatorname{LN}(X))
\]

\[
X''
=
X'+
\operatorname{MLP}(\operatorname{LN}(X'))
\]

要求：

- 使用 pre-norm；
- 保留两级残差；
- W-MSA 与 SW-MSA 交替；
- 使用多头注意力；
- 使用相对位置偏置；
- 使用 attention mask 处理移位窗口；
- 不将 SAR 物理先验直接加入 STL。

### 5.4 普通窗口与移位窗口

每个 RSTB 内部默认交替使用：

| STL 序号 | 注意力类型 | shift size |
|---|---|---:|
| 1 | W-MSA | 0 |
| 2 | SW-MSA | 4 |
| 3 | W-MSA | 0 |
| 4 | SW-MSA | 4 |
| 5 | W-MSA | 0 |
| 6 | SW-MSA | 4 |

必须实现并测试：

- window partition；
- window reverse；
- cyclic shift；
- reverse cyclic shift；
- attention mask；
- 非整除尺寸 padding；
- 推理后裁剪回原始尺寸。

### 5.5 RSTB 内部结构

第 \(i\) 个 RSTB：

\[
F_{i,L}
=
\operatorname{STL}_{i,L}
\left(
\cdots
\operatorname{STL}_{i,1}(F_{i,0})
\right)
\]

\[
\Delta F_i
=
H_{\mathrm{Conv}_i}(F_{i,L})
\]

\[
F_{i,\mathrm{out}}
=
F_{i,0}+\Delta F_i
\]

要求：

- 多个 STL 后使用一个 \(3\times3\) 卷积；
- 卷积输入输出通道均为 embedding dimension；
- 卷积结果与 RSTB 输入逐元素相加；
- 不替换为 \(1\times1\) 卷积；
- 不加入额外注意力或门控模块。

### 5.6 深层主干末尾卷积

多个 RSTB 后使用一个 \(3\times3\) 卷积：

\[
F_{\mathrm{DF}}
=
H_{\mathrm{Conv}}(F_K)
\]

再与浅层特征相加：

\[
F_{\mathrm{fusion}}
=
F_0+F_{\mathrm{DF}}
\]

要求：

- 保留该卷积；
- 不直接将最后一个 RSTB 输出与 \(F_0\) 相加；
- 不增加额外特征融合模块；
- 第一阶段不引入跨层 concat。

### 5.7 重建模块

第一阶段采用论文同尺寸恢复任务的单卷积重建头：

\[
R=H_{\mathrm{REC}}(F_{\mathrm{fusion}})
\]

默认输出通道数为 2，但必须由配置指定。

最终：

\[
\hat X_{\mathrm{out}}=X_{\mathrm{in}}+R
\]

第一阶段不使用：

- PixelShuffle；
- 反卷积；
- 插值上采样；
- 多尺度重建头；
- GAN 重建分支。

---

## 6. 与原论文的差异

| 项目 | 原论文 | 本项目 | 差异类型 |
|---|---|---|---|
| 主要任务 | 超分辨率、去噪、JPEG 去伪影 | PFA 散焦 SAR 图像重聚焦 | 必要差异 |
| 输入数据 | 灰度或 RGB 实值图像 | SAR 复图像的实值编码 | 必要差异 |
| 默认输入通道 | 1 或 3 | 暂定实部/虚部 2 通道 | 暂定差异 |
| 输出数据 | 高质量自然图像 | 聚焦 SAR 图像表示 | 必要差异 |
| 输入输出尺寸 | 视任务而定 | 第一阶段固定同尺寸 | 与论文同尺寸恢复分支一致 |
| 浅层卷积 | \(3\times3\) | \(3\times3\) | 一致 |
| RSTB 数量 | 标准配置 6 | 6 | 一致 |
| 每个 RSTB 的 STL 数量 | 标准配置 6 | 6 | 一致 |
| window size | 8 | 8 | 一致 |
| embedding dimension | 180 | 180 | 一致 |
| attention heads | 6 | 6 | 一致 |
| RSTB 内部卷积 | \(3\times3\) | \(3\times3\) | 一致 |
| 主干末尾卷积 | \(3\times3\) | \(3\times3\) | 一致 |
| 长残差连接 | 保留浅层特征 | 保留浅层 SAR 特征 | 结构一致，含义不同 |
| 重建头 | 同尺寸任务使用单卷积 | 单卷积 | 一致 |
| 图像级残差 | 输入加预测修正 | SAR 表示加预测修正 | 结构一致，数据含义不同 |
| 训练数据 | 自然图像退化对 | 同一回波的 PFA 与精确成像对 | 必要差异 |
| 相位对齐 | 不需要 | 需要 | SAR 特有 |
| 幅度归一化 | 自然图像规范 | SAR 复图像统一尺度 | SAR 特有 |
| 评价指标 | PSNR、SSIM | 复数 NMSE、幅度指标、聚焦指标等 | 必要差异 |
| 物理一致性 | 无 | 第一阶段不加入 | 暂缓 |
| 泛化设计 | 多自然图像数据集 | 后续研究跨轨迹、跨参数泛化 | 暂缓 |

---

## 7. 数据生成规范

### 7.1 样本生成流程

每个样本必须由同一份原始回波生成输入和标签。

#### 步骤 1：生成场景与回波

使用精确几何模型、实际轨迹和球面波距离生成原始回波：

\[
s(\tau,\eta)
\]

#### 步骤 2：生成散焦输入

使用当前 PFA 流程成像：

\[
X_{\mathrm{PFA}}
=
\mathcal I_{\mathrm{PFA}}(s)
\]

#### 步骤 3：生成聚焦标签

使用可信精确成像方法，例如：

- BP；
- 精确时域积分；
- 精确距离历史补偿；
- 其他已验证高精度算法。

得到：

\[
X_{\mathrm{GT}}
=
\mathcal I_{\mathrm{accurate}}(s)
\]

#### 步骤 4：形成训练对

\[
(X_{\mathrm{PFA}},X_{\mathrm{GT}})
\]

输入与标签必须源自：

- 同一场景；
- 同一原始回波；
- 同一坐标系；
- 同一成像区域；
- 同一像素网格。

### 7.2 坐标与空间配准要求

输入与标签必须保持一致：

- 场景中心；
- 方位向方向；
- 距离向方向；
- 坐标原点；
- 图像尺寸；
- 像素间距；
- 物理成像范围；
- FFT/IFT 符号约定；
- fftshift/ifftshift 使用方式。

验收要求：

\[
X_{\mathrm{PFA}}(i,j)
\quad\text{与}\quad
X_{\mathrm{GT}}(i,j)
\]

必须对应同一物理位置。

禁止让模型同时学习：

- 图像平移；
- 图像旋转；
- 尺度变化；
- 坐标轴翻转；
- 额外插值误差。

### 7.3 全局相位对齐

输入和标签可能存在全局相位偏差：

\[
X_{\mathrm{GT}}
\approx
X_{\mathrm{PFA}}e^{j\phi_0}
\]

可通过以下方式估计：

\[
\hat\phi_0
=
\arg
\left(
\sum_{x,y}
X_{\mathrm{GT}}(x,y)
X_{\mathrm{PFA}}^*(x,y)
\right)
\]

然后统一相位参考。

要求：

- 全局相位对齐逻辑必须独立实现；
- 可通过配置开关控制；
- 对齐前后均保存统计信息；
- 不允许每个像素独立对齐；
- 不允许利用标签进行空间变化相位校正。

全局相位对齐的目的只是消除无物理意义的整体相位旋转。

### 7.4 幅度归一化

输入和标签必须使用同一个缩放因子：

\[
\tilde X_{\mathrm{PFA}}
=
\frac{X_{\mathrm{PFA}}}{s}
\]

\[
\tilde X_{\mathrm{GT}}
=
\frac{X_{\mathrm{GT}}}{s}
\]

禁止输入和标签分别独立归一化。

可选策略：

1. 全训练集统一固定缩放；
2. 每个样本共享缩放因子；
3. 根据输入统计量确定缩放，并对标签使用同一因子。

必须保存缩放因子，以便推理结果恢复原始量纲。

### 7.5 数据切块

训练 patch 要求：

- 高和宽为 window size 的整数倍；
- 输入与标签使用完全相同的裁剪位置；
- patch 能完整覆盖典型散焦响应；
- 不应将主瓣与主要拖尾切断；
- 不应让同一目标跨训练/验证/测试集合。

候选 patch size：

- 64；
- 128；
- 256。

最终 patch size 由最大典型散焦范围决定，而不是机械照搬自然图像设置。

### 7.6 数据划分

必须先按完整场景划分，再生成 patch。

默认：

- 训练集：70%；
- 验证集：15%；
- 测试集：15%。

同一原始场景、同一回波或其任何裁剪结果只能属于一个集合。

第一阶段允许训练集和测试集共享相似的：

- 雷达参数分布；
- 轨迹误差分布；
- 场景类型；
- 目标数量范围；
- 信噪比范围。

此时评估的是同分布重聚焦能力。

---

## 8. 损失函数规范

### 8.1 第一版默认损失

采用 Charbonnier 损失：

\[
\mathcal L_{\mathrm{char}}
=
\sqrt{
\|\hat X-X_{\mathrm{GT}}\|_2^2+\epsilon^2
}
\]

默认：

\[
\epsilon=10^{-3}
\]

当输入输出为实部/虚部双通道时，两个通道共同参与损失计算。

建议实现为逐元素 Charbonnier 后取均值：

\[
\mathcal L
=
\frac{1}{N}
\sum_n
\sqrt{
(\hat x_n-x_n)^2+\epsilon^2
}
\]

### 8.2 第一阶段不加入的损失

第一阶段禁止默认加入：

- GAN loss；
- perceptual loss；
- 相位专用 loss；
- SAR 图像熵 loss；
- 梯度 loss；
- 稀疏正则；
- 回波域数据一致性；
- 显式相位误差正则；
- 轨迹参数监督。

这些内容只能在原始基线完成后作为独立实验引入。

---

## 9. 训练配置

以下超参数必须配置化：

```yaml
model:
  in_channels: 2
  out_channels: 2
  embed_dim: 180
  depths: [6, 6, 6, 6, 6, 6]
  num_heads: [6, 6, 6, 6, 6, 6]
  window_size: 8
  mlp_ratio: 2.0
  residual_connection: "1conv"
  use_image_residual: true

data:
  input_representation: "real_imag"
  output_representation: "real_imag"
  patch_size: 128
  phase_alignment: true
  normalization: "shared_sample_scale"

loss:
  type: "charbonnier"
  epsilon: 1.0e-3

train:
  optimizer: "adam"
  learning_rate: null
  batch_size: null
  total_iterations: null
  scheduler: null
  seed: 42
```

其中：

- `in_channels`、`out_channels` 暂定为 2；
- 输入输出表示必须独立配置；
- 学习率、batch size、总迭代次数和 scheduler 在核验原论文官方配置后确定；
- 禁止在训练脚本中分散写死关键超参数。

---

## 10. 实现目录建议

```text
swinir_sar/
├── configs/
│   ├── swinir_sar_baseline.yaml
│   └── swinir_original_reference.yaml
├── datasets/
│   ├── sar_pair_dataset.py
│   ├── transforms.py
│   ├── normalization.py
│   └── phase_alignment.py
├── models/
│   ├── swinir.py
│   ├── rstb.py
│   ├── swin_transformer_layer.py
│   ├── window_attention.py
│   └── common.py
├── losses/
│   └── charbonnier.py
├── metrics/
│   ├── complex_nmse.py
│   ├── amplitude_metrics.py
│   ├── phase_metrics.py
│   └── focusing_metrics.py
├── scripts/
│   ├── generate_pairs.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── tests/
│   ├── test_window_ops.py
│   ├── test_shapes.py
│   ├── test_residual_path.py
│   ├── test_phase_alignment.py
│   └── test_dataset_pairing.py
├── outputs/
└── README.md
```

---

## 11. 单元测试与正确性验证

### 11.1 Window 操作测试

验证：

\[
\operatorname{WindowReverse}
\left(
\operatorname{WindowPartition}(X)
\right)
=
X
\]

测试内容：

- 可整除尺寸；
- 不可整除尺寸；
- padding 后恢复；
- shift 后恢复；
- 多 batch；
- 多 channel。

### 11.2 张量尺寸测试

输入：

\[
[B,C_{\mathrm{in}},H,W]
\]

默认：

\[
[B,2,H,W]
\]

输出必须为：

\[
[B,C_{\mathrm{out}},H,W]
\]

默认：

\[
[B,2,H,W]
\]

中间张量应满足：

- 浅层特征：\([B,180,H,W]\)；
- 每个 RSTB 输出：\([B,180,H,W]\)；
- 主干末尾卷积：\([B,180,H,W]\)；
- 重建输出：\([B,2,H,W]\)。

### 11.3 图像级残差测试

当预测残差分支输出为零时：

\[
\hat X=X_{\mathrm{in}}
\]

该测试用于确认图像级残差实现正确。

### 11.4 参数量核验

修改输入输出通道后，参数量只应在第一层和最后一层发生少量变化。

如果与原始标准 SwinIR 参数量差异明显，应检查：

- RSTB 数量；
- STL 数量；
- embedding dimension；
- QKV 投影；
- MLP ratio；
- attention heads；
- RSTB 卷积；
- 主干末尾卷积。

### 11.5 小样本过拟合测试

使用极少量样本，例如 4～16 个场景，验证模型能够显著降低训练误差。

通过条件：

- loss 持续下降；
- 输出逐渐接近 GT；
- 无 NaN/Inf；
- 幅度和相位结果无明显异常；
- 能明显改善训练样本的聚焦质量。

如果小样本无法过拟合，不得直接扩大训练规模。

---

## 12. 评价指标

### 12.1 复数域指标

#### 复数 NMSE

\[
\mathrm{NMSE}
=
\frac{
\|\hat X-X_{\mathrm{GT}}\|_2^2
}{
\|X_{\mathrm{GT}}\|_2^2
}
\]

该指标为第一阶段主要整体误差指标。

### 12.2 幅度域指标

对：

\[
|\hat X|
\quad\text{与}\quad
|X_{\mathrm{GT}}|
\]

计算：

- PSNR；
- SSIM；
- MAE；
- 峰值幅度误差；
- 对数幅度误差。

### 12.3 相位指标

在有效散射区域内计算：

\[
\Delta\phi
=
\arg
\left(
\hat X X_{\mathrm{GT}}^*
\right)
\]

建议仅在以下区域统计：

\[
|X_{\mathrm{GT}}|>\tau
\]

避免背景低幅度区域随机相位主导结果。

### 12.4 聚焦指标

对于点目标或孤立散射中心，计算：

- IRW；
- PSLR；
- ISLR；
- 峰值位置误差；
- 峰值幅度误差；
- 距离向剖面；
- 方位向剖面。

禁止只依据“图像看起来更清晰”判断模型是否完成重聚焦。

---

## 13. 实验输出要求

每个测试样本至少保存：

1. PFA 散焦输入的幅度图；
2. PFA 散焦输入的相位图；
3. SwinIR 输出幅度图；
4. SwinIR 输出相位图；
5. 精确聚焦 GT 幅度图；
6. 精确聚焦 GT 相位图；
7. 输出与 GT 的幅度误差图；
8. 输出与 GT 的相位误差图；
9. 典型目标距离向剖面；
10. 典型目标方位向剖面。

同时保存：

- 样本 ID；
- 场景参数；
- 雷达参数；
- 轨迹参数；
- PFA 参数；
- GT 成像参数；
- 归一化因子；
- 全局相位对齐量。

---

## 14. 里程碑

### M1：原始 SwinIR 主体复现

完成：

- Window Attention；
- W-MSA；
- SW-MSA；
- STL；
- RSTB；
- 深层主干；
- 长残差；
- 同尺寸重建头；
- Charbonnier 损失。

验收：

- 结构测试通过；
- 参数量合理；
- 输入输出尺寸正确；
- 前向、反向传播正常。

### M2：SAR 数据链路

完成：

- 精确回波生成；
- PFA 散焦图像生成；
- 精确聚焦 GT 生成；
- 配准；
- 相位对齐；
- 统一归一化；
- 数据切块；
- 数据集划分。

验收：

- 输入与标签物理位置严格一致；
- 输入和标签来自同一份回波；
- 不存在独立归一化；
- 不存在 train/val/test 场景泄漏。

### M3：小样本过拟合

验收：

- 少量样本可以显著过拟合；
- 复数 NMSE 明显下降；
- 输出幅度与相位逐渐接近 GT；
- 点目标聚焦指标改善。

### M4：同分布正式实验

固定：

- 载频范围；
- 带宽；
- 孔径；
- 成像网格；
- 轨迹扰动范围；
- 场景类型；
- 目标数量范围；
- PFA 实现版本；
- GT 算法版本。

验收：

- 测试集复数 NMSE 优于输入；
- IRW 改善；
- PSLR/ISLR 改善；
- 目标峰值位置基本稳定；
- 不出现明显虚假强散射中心。

### M5：输入输出表示消融

在主基线完成后比较：

- 实部/虚部；
- 幅度/相位；
- 对数幅度/\(\cos\phi\)/\(\sin\phi\)；
- 其他多通道组合。

该阶段不改变 SwinIR 中间主干。

### M6：泛化研究

后续逐步测试：

- 未见轨迹；
- 未见近场距离；
- 不同载频；
- 不同带宽；
- 不同孔径；
- 不同目标密度；
- 不同信噪比；
- 不同场景类型；
- 仿真到实测迁移。

泛化研究不属于第一阶段基线验收范围。

---

## 15. 风险与控制措施

### 风险 1：输入和标签不严格配准

后果：模型学习配准而不是重聚焦。

控制：

- 统一坐标网格；
- 输出样本配准可视化；
- 使用点目标位置自动校验。

### 风险 2：全局相位参考不一致

后果：网络学习无意义的整体相位旋转。

控制：

- 使用统一参考距离；
- 实现全局相位对齐；
- 保存对齐相位量。

### 风险 3：分别归一化输入和标签

后果：真实散射强度关系被破坏。

控制：

- 输入和标签共享缩放因子；
- 保存缩放元数据。

### 风险 4：patch 太小

后果：散焦能量被裁断，网络无法学习完整回聚过程。

控制：

- 统计最大散焦范围；
- patch size 大于典型响应范围；
- 必要时使用整图训练或重叠 patch。

### 风险 5：只看视觉清晰度

后果：输出看似锐利，但目标位置、幅度、旁瓣不可信。

控制：

- 强制记录复数 NMSE；
- 强制计算 IRW、PSLR、ISLR；
- 保存距离向和方位向剖面。

### 风险 6：输入输出形式过早固定

后果：后续发现实部/虚部并非最佳表示时，接口难以修改。

控制：

- 输入输出通道完全配置化；
- 表示转换放在 dataset/transform 层；
- 模型主干只接收实值张量，不绑定物理表示。

---

## 16. 第一阶段完成定义（Definition of Done）

满足以下全部条件，第一阶段基线视为完成：

- [ ] SwinIR 主体结构按原论文实现；
- [ ] 输入输出通道可配置；
- [ ] 第一版支持实部/虚部双通道；
- [ ] 同尺寸恢复头实现完成；
- [ ] 图像级残差可配置；
- [ ] Window Attention 单元测试通过；
- [ ] 输入输出尺寸测试通过；
- [ ] 数据配准检查通过；
- [ ] 全局相位对齐实现完成；
- [ ] 输入与标签共享归一化；
- [ ] 小样本过拟合成功；
- [ ] 完成同分布训练与测试；
- [ ] 输出复数 NMSE；
- [ ] 输出幅度 PSNR/SSIM；
- [ ] 输出相位误差；
- [ ] 输出 IRW、PSLR、ISLR；
- [ ] 保存输入、输出、GT 与误差可视化；
- [ ] 实验配置、随机种子和数据版本可复现；
- [ ] 未在第一阶段私自加入额外物理模块、GAN 或复杂损失。

---

## 17. 第一版默认基线配置

\[
\boxed{
\begin{aligned}
\text{任务}&:\ \text{PFA散焦SAR图像同尺寸重聚焦}\\
\text{输入}&:\ [\operatorname{Re}(X_{\mathrm{PFA}}),\operatorname{Im}(X_{\mathrm{PFA}})]\ \text{（暂定）}\\
\text{主干}&:\ \text{标准SwinIR}\\
\text{RSTB数量}&:\ 6\\
\text{每个RSTB的STL数量}&:\ 6\\
\text{window size}&:\ 8\\
\text{shift size}&:\ 4\\
\text{embedding dimension}&:\ 180\\
\text{attention heads}&:\ 6\\
\text{输出}&:\ [\operatorname{Re}(X_{\mathrm{focus}}),\operatorname{Im}(X_{\mathrm{focus}})]\ \text{（暂定）}\\
\text{空间尺度}&:\ 1\times\\
\text{重建头}&:\ \text{单卷积同尺寸重建}\\
\text{残差形式}&:\ \hat X=X_{\mathrm{in}}+R\\
\text{损失}&:\ \text{Charbonnier},\ \epsilon=10^{-3}
\end{aligned}
}
\]

该配置的目标不是证明其已经是最优方案，而是建立一个结构忠实、任务边界清晰、可重复验证的 SwinIR-SAR 重聚焦基线。
