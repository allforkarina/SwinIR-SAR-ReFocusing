# SwinIR-SAR 重聚焦实施计划（Implementation Plan）

> 版本：v0.1  
> 依据：`SwinIR-SAR_重聚焦实施规范_v0.1.md`  
> 目标：在不改变 SwinIR 中间主干设计的前提下，完成一个可训练、可评估、可复现的 SAR 图像重聚焦基线  
> 第一版预设输入输出：实部/虚部双通道；该设计保持可配置，不作为最终结论

---

## 1. 总体目标

完成以下完整闭环：

\[
\text{精确回波生成}
\rightarrow
\text{PFA散焦成像}
\rightarrow
\text{精确聚焦GT}
\rightarrow
\text{SwinIR训练}
\rightarrow
\text{重聚焦输出}
\rightarrow
\text{SAR指标评价}
\]

最终需要回答：

> 在固定或近似同分布的雷达参数、轨迹误差和场景条件下，标准 SwinIR 是否能够学习从 PFA 散焦复图像到聚焦复图像的映射？

---

# 2. 实施原则

1. **先复现，再适配，再训练。**
2. **模型主干严格遵循原论文。**
3. **输入输出形式配置化。**
4. **第一版只做同尺寸 refocusing。**
5. **先完成小样本过拟合，再扩大训练。**
6. **先验证同分布能力，再研究泛化。**
7. **禁止第一阶段同时加入物理展开、GAN、复杂损失和轨迹编码。**
8. **所有实验必须可复现、可回滚、可比较。**

---

# 3. 阶段划分

整个实施过程分为 8 个阶段：

| 阶段 | 名称 | 核心目标 |
|---|---|---|
| P0 | 环境与仓库初始化 | 建立稳定、可复现的开发环境 |
| P1 | 原始 SwinIR 结构复现 | 完成模型主体并通过单元测试 |
| P2 | 原论文任务 sanity check | 确认模型实现本身正确 |
| P3 | SAR 数据生成链路 | 生成 PFA 输入与精确聚焦 GT |
| P4 | SAR 数据适配 | 完成复数表示、归一化、配准和数据集 |
| P5 | 小样本过拟合 | 验证模型能够学习重聚焦映射 |
| P6 | 同分布正式训练 | 建立第一版可比较基线 |
| P7 | 评估、复盘与归档 | 输出结论、失败分析和下一阶段依据 |

---

# 4. P0：环境与仓库初始化

## 4.1 任务

- 创建独立项目仓库；
- 建立 Python 虚拟环境；
- 固定 PyTorch、CUDA 和关键依赖版本；
- 建立基础目录；
- 配置日志、随机种子和实验输出路径；
- 添加 Git 版本管理；
- 创建配置文件模板；
- 保存当前 PFA 与精确成像代码版本。

## 4.2 推荐目录

```text
swinir_sar/
├── configs/
├── datasets/
├── models/
├── losses/
├── metrics/
├── scripts/
├── tests/
├── tools/
├── outputs/
├── docs/
├── requirements.txt
├── train.py
├── evaluate.py
└── README.md
```

## 4.3 输出物

- 可运行的 Python 环境；
- `requirements.txt` 或 `environment.yml`；
- 项目目录；
- 基础配置文件；
- README；
- 随机种子控制函数；
- 当前 PFA、BP 或精确成像代码版本记录。

## 4.4 验收条件

- 可以成功导入 PyTorch；
- GPU 前向和反向测试通过；
- 同一随机种子运行结果可重复；
- 实验配置可以完整保存；
- 不依赖未记录的本地路径。

---

# 5. P1：原始 SwinIR 结构复现

## 5.1 实施顺序

建议按由底向上的顺序实现。

### 步骤 1：基础工具函数

实现：

- `to_2tuple`；
- stochastic depth / DropPath；
- trunc normal 初始化；
- padding 与裁剪；
- tensor shape 转换。

### 步骤 2：窗口操作

实现：

- `window_partition`；
- `window_reverse`；
- cyclic shift；
- reverse shift；
- attention mask 生成。

### 步骤 3：WindowAttention

实现：

- QKV 线性映射；
- 多头拆分；
- scaled dot-product attention；
- relative position bias；
- mask；
- softmax；
- 输出投影。

### 步骤 4：Swin Transformer Layer

实现：

- LayerNorm；
- W-MSA / SW-MSA；
- 第一条残差；
- MLP；
- 第二条残差；
- 动态输入尺寸支持。

### 步骤 5：RSTB

实现：

- 多个 STL；
- feature reshape；
- \(3\times3\) 卷积；
- RSTB 残差连接。

### 步骤 6：完整 SwinIR

实现：

- 浅层卷积；
- 6 个 RSTB；
- 深层主干末尾卷积；
- 长残差连接；
- 同尺寸单卷积重建头；
- 可选图像级残差。

## 5.2 默认模型参数

```yaml
embed_dim: 180
depths: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
window_size: 8
mlp_ratio: 2.0
residual_connection: "1conv"
```

## 5.3 单元测试

必须完成：

### 窗口可逆性

\[
\operatorname{reverse}(\operatorname{partition}(X))=X
\]

### 移位可逆性

shift 后 reverse shift 恢复原特征。

### 尺寸测试

输入：

\[
[B,C,H,W]
\]

输出：

\[
[B,C_{\text{out}},H,W]
\]

### 非整除尺寸测试

验证 \(H,W\) 不能被 window size 整除时：

- padding 正确；
- attention 正确；
- 输出正确裁剪。

### 梯度测试

确认：

- 无 NaN；
- 无 Inf；
- 参数梯度非空；
- 反向传播正常。

### 参数量测试

与原论文标准模型保持同一数量级。输入输出通道变化只允许影响首尾卷积。

## 5.4 输出物

- `models/window_attention.py`
- `models/swin_transformer_layer.py`
- `models/rstb.py`
- `models/swinir.py`
- 对应单元测试
- 模型参数统计脚本

## 5.5 验收条件

- 全部结构测试通过；
- 任意合法尺寸可前向；
- 可以反向传播；
- 模型参数量合理；
- 中间主干与论文设计一致。

---

# 6. P2：原论文任务 Sanity Check

这一阶段不是为了取得论文同等性能，而是排除模型主体实现错误。

## 6.1 任务

选择一个简单的自然图像同尺寸恢复任务，例如：

- 高斯去噪；
- 简单模糊恢复；
- JPEG 去伪影中的简化版本。

构造小规模自然图像数据：

\[
I_{\mathrm{LQ}}\rightarrow I_{\mathrm{HQ}}
\]

使用原论文同尺寸恢复头和 Charbonnier 损失训练。

## 6.2 目的

确认以下模块正确：

- SwinIR 主干；
- Window Attention；
- RSTB；
- 重建头；
- 残差；
- 损失；
- 优化器；
- checkpoint；
- 推理脚本。

## 6.3 输出物

- 小规模自然图像实验；
- loss 曲线；
- 输入、输出和 GT 对比；
- checkpoint；
- 推理结果。

## 6.4 验收条件

- 网络可以过拟合少量自然图像；
- 输出质量明显优于输入；
- loss 稳定下降；
- 训练和推理流程完整。

若该阶段失败，不进入 SAR 数据训练。

---

# 7. P3：SAR 数据生成链路

## 7.1 场景生成

第一版建议从可控场景开始：

### 数据层级 A：单点目标

用于验证：

- 主瓣；
- 旁瓣；
- 峰值位置；
- 距离向与方位向聚焦。

### 数据层级 B：多点目标

用于验证：

- 多目标可分辨性；
- 相邻目标干扰；
- 不同位置空间变化散焦。

### 数据层级 C：稀疏扩展场景

用于验证：

- 复杂结构；
- 多散射中心；
- 局部纹理和轮廓。

第一版正式训练可先使用 A+B，再逐步加入 C。

## 7.2 回波生成

对每个场景保存：

- 场景反射率；
- 雷达参数；
- 轨迹；
- 目标坐标；
- 原始回波；
- 随机噪声参数；
- 生成脚本版本。

## 7.3 散焦输入生成

使用当前 PFA 处理同一份回波：

\[
X_{\mathrm{PFA}}
=
\mathcal I_{\mathrm{PFA}}(s)
\]

必须固定：

- PFA 代码版本；
- 极坐标重采样方式；
- 场景中心；
- \(k_x,k_y\) 网格；
- FFT 方向；
- 成像范围。

## 7.4 聚焦 GT 生成

使用精确算法生成：

\[
X_{\mathrm{GT}}
=
\mathcal I_{\mathrm{accurate}}(s)
\]

推荐优先级：

1. 精确 BP；
2. 已验证的精确时域积分；
3. 其他具有可信几何补偿的算法。

## 7.5 数据格式

建议每个样本保存为：

```text
sample_xxxxxx.npz
├── pfa_real
├── pfa_imag
├── gt_real
├── gt_imag
├── scene_metadata
├── radar_metadata
├── trajectory_metadata
├── normalization_scale
└── global_phase_offset
```

也可使用 HDF5 或 Zarr，但必须支持：

- 分块读取；
- 元数据保存；
- 版本追踪；
- 高效训练。

## 7.6 验收条件

- 输入与 GT 来自同一份回波；
- 输入和 GT 尺寸一致；
- 坐标一致；
- 点目标位置一致；
- 差异主要表现为散焦，而不是平移或旋转；
- GT 聚焦指标明显优于 PFA 输入。

---

# 8. P4：SAR 数据适配

## 8.1 输入输出表示

第一版预设：

```yaml
input_representation: real_imag
output_representation: real_imag
in_channels: 2
out_channels: 2
```

但代码必须允许替换为：

- amplitude_phase；
- logamplitude_phase；
- amplitude_cos_sin；
- 多通道混合表示。

## 8.2 全局相位对齐

实现：

\[
\hat\phi_0
=
\arg
\left(
\sum X_{\mathrm{GT}}X_{\mathrm{PFA}}^*
\right)
\]

记录：

- 对齐前相位差；
- 对齐后相位差；
- 对齐角；
- 是否启用对齐。

## 8.3 归一化

第一版推荐优先比较两种策略：

### 策略 A：全训练集统一尺度

适合保留跨样本绝对强度关系。

### 策略 B：单样本共享尺度

输入与 GT 使用同一个缩放值。

禁止：

- 输入独立归一化；
- GT 独立归一化。

## 8.4 Patch 生成

实施步骤：

1. 统计散焦响应的最大范围；
2. 选择 64、128 或 256；
3. 保证 patch size 为 8 的整数倍；
4. 输入和 GT 同位置裁剪；
5. 对点目标优先保证主瓣和主要拖尾完整；
6. 场景划分后再切 patch。

## 8.5 数据增强

第一版只允许不会破坏复数物理关系的联合增强：

- 水平翻转；
- 垂直翻转；
- 90° 旋转。

前提是：

- 输入与 GT 同步变换；
- 坐标方向变化不会破坏评价解释。

第一版可先完全关闭数据增强，减少不确定性。

## 8.6 验收条件

- dataset 输出形状正确；
- 输入和 GT 使用同一缩放；
- 无 NaN/Inf；
- phase alignment 可复现；
- patch 对齐；
- train/val/test 无场景泄漏。

---

# 9. P5：小样本过拟合

## 9.1 目标

验证模型是否能够学习当前任务，而非验证泛化。

## 9.2 数据规模

建议使用：

- 4～16 个完整场景；
- 或几十个严格配对 patch。

## 9.3 训练设置

- 关闭数据增强；
- 固定随机种子；
- 使用较小 batch；
- 训练足够长时间；
- 每隔固定步数保存结果；
- 同时保存输入、输出、GT。

## 9.4 观察指标

- Charbonnier loss；
- 复数 NMSE；
- 幅度 PSNR；
- 幅度 SSIM；
- 相位误差；
- IRW；
- PSLR；
- ISLR；
- 峰值位置。

## 9.5 通过标准

至少满足：

- 训练 loss 明显下降；
- 输出复数 NMSE 显著优于 PFA 输入；
- 点目标主瓣变窄；
- 峰值位置不明显漂移；
- 输出逐渐接近 GT；
- 无严重伪目标。

## 9.6 失败排查顺序

若无法过拟合，依次检查：

1. 输入与 GT 是否配准；
2. 全局相位是否一致；
3. 归一化是否共享；
4. 残差加法是否正确；
5. 输入输出通道是否匹配；
6. loss 是否按双通道正确计算；
7. patch 是否切断散焦响应；
8. 模型是否实际更新；
9. GT 是否真的比 PFA 聚焦；
10. 是否存在 FFT 方向或坐标翻转错误。

---

# 10. P6：同分布正式训练

## 10.1 数据划分

按完整场景划分：

- 训练集：70%；
- 验证集：15%；
- 测试集：15%。

同一场景的所有 patch 只能进入一个集合。

## 10.2 基线实验

第一版至少完成以下三组对比：

### B0：PFA 输入

不经过模型，作为原始基线。

### B1：SwinIR-SAR

标准 SwinIR 主干，实部/虚部输入输出。

### B2：简单 CNN 基线

使用简单残差 CNN，参数量尽量控制在合理范围。

目的：判断效果来自 SwinIR 的窗口注意力，还是普通深层卷积也能实现。

## 10.3 训练记录

每次实验必须保存：

- Git commit；
- 配置文件；
- 数据版本；
- 随机种子；
- 模型参数量；
- GPU 信息；
- batch size；
- 学习率；
- scheduler；
- 总迭代数；
- 最佳 checkpoint；
- 最后 checkpoint；
- 日志；
- 评价结果；
- 可视化结果。

## 10.4 模型选择

以验证集主指标选择最佳模型。

第一版推荐主指标：

\[
\mathrm{NMSE}_{\mathrm{complex}}
\]

辅助指标：

- 幅度 PSNR；
- IRW；
- PSLR；
- ISLR。

## 10.5 通过标准

SwinIR-SAR 相比 PFA 输入至少在以下多数指标上改善：

- 复数 NMSE；
- 幅度 PSNR；
- IRW；
- PSLR；
- ISLR。

同时要求：

- 峰值位置基本保持；
- 不明显制造伪目标；
- 多目标可分辨性不下降。

---

# 11. P7：评估与实验归档

## 11.1 定量评估

测试集汇总：

- mean；
- median；
- std；
- best；
- worst。

对指标：

- complex NMSE；
- amplitude PSNR；
- amplitude SSIM；
- phase MAE；
- IRW；
- PSLR；
- ISLR；
- peak location error；
- peak amplitude error。

## 11.2 定性评估

每类场景至少展示：

- 单点目标；
- 多点目标；
- 边缘区域目标；
- 场景中心目标；
- 严重散焦样本；
- 轻度散焦样本；
- 最佳样本；
- 最差样本。

每个样本输出：

1. PFA 幅度；
2. PFA 相位；
3. 模型输出幅度；
4. 模型输出相位；
5. GT 幅度；
6. GT 相位；
7. 幅度误差图；
8. 相位误差图；
9. 距离向剖面；
10. 方位向剖面。

## 11.3 结论分类

最终结果按以下三类归纳：

### 结论 A：模型能够完成同分布重聚焦

说明 SwinIR 主干具备学习 PFA 散焦映射的能力。

### 结论 B：幅度变锐利，但复数指标和聚焦指标未改善

说明模型主要进行了视觉锐化，未真正恢复相干结构。

### 结论 C：模型无法稳定学习

需要回到：

- 数据对齐；
- 输入表示；
- GT；
- 归一化；
- patch；
- 任务可逆性；

逐项排查，而不是立即增加更复杂模型。

---

# 12. 推荐时间安排

以下为一个 4 周实施周期，可根据算力和数据生成速度调整。

## 第 1 周：模型复现

### Day 1

- 初始化仓库；
- 固定环境；
- 创建配置系统；
- 实现基础工具。

### Day 2

- 实现 window partition/reverse；
- 实现 relative position index；
- 完成窗口单元测试。

### Day 3

- 实现 WindowAttention；
- 实现 mask；
- 完成 attention shape 测试。

### Day 4

- 实现 STL；
- 实现 W-MSA/SW-MSA；
- 验证前向和反向。

### Day 5

- 实现 RSTB；
- 实现完整 SwinIR；
- 完成参数量和尺寸测试。

### Day 6～7

- 自然图像小任务 sanity check；
- 修复模型实现问题。

## 第 2 周：SAR 数据链路

### Day 8～9

- 整理场景与轨迹生成代码；
- 固定元数据格式；
- 生成单点目标数据。

### Day 10～11

- 生成 PFA 输入；
- 生成 BP/精确 GT；
- 检查坐标和峰值位置。

### Day 12

- 实现全局相位对齐；
- 实现共享归一化。

### Day 13

- 实现 SAR pair dataset；
- 实现 patch；
- 完成数据单元测试。

### Day 14

- 批量生成小规模数据；
- 人工和定量检查样本质量。

## 第 3 周：小样本验证与正式训练

### Day 15～16

- 小样本过拟合；
- 检查复数 NMSE；
- 检查聚焦剖面。

### Day 17

- 修复数据、残差或归一化问题；
- 固化第一版配置。

### Day 18～20

- 生成正式训练集；
- 运行 SwinIR-SAR 基线；
- 运行简单 CNN 对照。

### Day 21

- 选择最佳 checkpoint；
- 运行测试集评估。

## 第 4 周：评估与归档

### Day 22～23

- 完成复数、幅度、相位指标；
- 完成 IRW/PSLR/ISLR。

### Day 24～25

- 生成定性图；
- 整理最佳和最差案例；
- 分析伪目标与失效模式。

### Day 26

- 复现实验；
- 验证固定随机种子和配置可重跑。

### Day 27～28

- 整理阶段报告；
- 总结是否可行；
- 制定下一阶段输入表示与泛化实验。

---

# 13. 优先级排序

## P0：必须首先完成

- 数据配准；
- 全局相位参考；
- 共享归一化；
- SwinIR 结构正确；
- 小样本过拟合。

## P1：正式基线必须完成

- 同分布训练；
- 复数 NMSE；
- IRW/PSLR/ISLR；
- CNN 对照；
- 结果可视化。

## P2：主基线完成后再做

- 幅度/相位输入；
- \(\cos\phi,\sin\phi\) 编码；
- patch size 对照；
- 模型深度对照；
- 输入表示消融。

## P3：后续研究

- 跨轨迹泛化；
- 跨距离泛化；
- 跨载频/带宽；
- 仿真到实测；
- 物理一致性；
- 相位误差显式预测；
- 深度展开。

---

# 14. 第一版配置冻结点

当满足以下条件后，冻结第一版基线配置：

- SwinIR 自然图像 sanity check 通过；
- SAR 数据配准通过；
- 小样本可过拟合；
- 实部/虚部双通道可以稳定训练；
- loss 与评价脚本无明显错误。

冻结后，以下内容不得在同一基线实验中随意变更：

- 模型结构；
- 输入表示；
- 输出表示；
- 归一化策略；
- 相位对齐方式；
- patch size；
- 训练集划分；
- 损失函数；
- 评价指标。

任何修改必须建立新的实验编号。

---

# 15. 实验编号建议

```text
EXP-SAR-SWINIR-001
```

推荐命名：

```text
EXP-SAR-SWINIR-001_realimag_baseline
EXP-SAR-SWINIR-002_realimag_patch256
EXP-SAR-SWINIR-003_amp_phase
EXP-SAR-CNN-001_realimag_baseline
```

每个实验目录保存：

```text
experiment/
├── config.yaml
├── train.log
├── metrics.json
├── best.pth
├── last.pth
├── visualizations/
├── curves/
└── environment.txt
```

---

# 16. 最终交付物

第一阶段结束时应至少具备：

- [ ] 可运行的 SwinIR-SAR 代码；
- [ ] 原论文主体结构实现；
- [ ] 完整单元测试；
- [ ] SAR 配对数据生成脚本；
- [ ] PFA 输入与精确 GT 数据集；
- [ ] 全局相位对齐模块；
- [ ] 共享归一化模块；
- [ ] 训练脚本；
- [ ] 推理脚本；
- [ ] 评价脚本；
- [ ] 小样本过拟合结果；
- [ ] 同分布正式实验；
- [ ] CNN 对照实验；
- [ ] 复数与聚焦指标；
- [ ] 可视化结果；
- [ ] 配置、日志和 checkpoint；
- [ ] 第一阶段实验报告；
- [ ] 后续泛化研究问题清单。

---

# 17. 最短执行路径

若希望尽快得到第一批可验证结果，可以采用以下最短路径：

1. 复现同尺寸 SwinIR；
2. 只生成单点和多点目标；
3. 使用 PFA 生成散焦输入；
4. 使用 BP 生成聚焦 GT；
5. 输入输出暂用实部/虚部；
6. 统一相位和尺度；
7. 使用 128 或 256 patch；
8. 进行 8～16 个场景的小样本过拟合；
9. 计算复数 NMSE、IRW、PSLR 和 ISLR；
10. 判断模型是否真正完成重聚焦。

该路径的优点是变量少、结论明确，可以尽早发现模型、数据或任务定义上的根本问题。
