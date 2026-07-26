# SwinIR 独立架构复现实施规范（Implementation Spec）

> 版本：v0.4  
> 当前阶段：从零实现并验证 SwinIR 网络架构  
> 运行时依赖：PyTorch；不依赖官方 SwinIR 仓库  
> 参考基准：SwinIR 论文与官方 `network_swinir.py`  
> 后续任务预设：SAR 实部/虚部双通道、同尺寸重聚焦；当前不需要真实输入数据

---

## 1. 目标与边界

本阶段的目标是使用 PyTorch 基础组件独立实现 SwinIR，而不是在项目中直接导入官方 `network_swinir.py`。

需要实现：

- SwinIR 所需的基础工具；
- 窗口划分与恢复；
- 窗口多头自注意力；
- 普通窗口与移位窗口；
- Swin Transformer Block；
- BasicLayer；
- PatchEmbed 与 PatchUnEmbed；
- RSTB；
- SwinIR 主网络；
- 当前 SAR 任务需要的同尺寸恢复分支；
- 与官方实现的结构、参数量和数值等价验证。

当前阶段不包含：

- SAR 数据生成；
- Dataset；
- 正式损失设计；
- 优化器与正式训练；
- 重聚焦效果评价；
- 泛化实验。

---

## 2. 官方仓库的使用方式

项目运行时不需要 clone 或导入官方仓库。

但为了保证复现准确，建议保存一份固定版本的参考材料：

```text
references/
├── network_swinir.py
├── main_test_swinir.py
├── LICENSE
├── README.md
└── UPSTREAM_COMMIT.txt
```

这些文件只用于：

- 对照模块定义；
- 核对默认参数；
- 核对张量路径；
- 核对权重命名；
- 进行数值等价测试；
- 保留开源许可与来源说明。

不得在自己的模型代码中执行：

```python
from references.network_swinir import SwinIR
```

自己的实现必须位于独立源码目录。

---

## 3. 项目目录规范

```text
SwinIR-SAR/
├── references/
│   ├── network_swinir.py
│   ├── main_test_swinir.py
│   ├── LICENSE
│   └── UPSTREAM_COMMIT.txt
├── swinir/
│   ├── __init__.py
│   ├── common.py
│   ├── mlp.py
│   ├── window_ops.py
│   ├── window_attention.py
│   ├── swin_block.py
│   ├── basic_layer.py
│   ├── patch_ops.py
│   ├── rstb.py
│   ├── upsample.py
│   └── model.py
├── configs/
│   └── swinir_same_size.yaml
├── tests/
│   ├── test_window_ops.py
│   ├── test_attention.py
│   ├── test_swin_block.py
│   ├── test_rstb.py
│   ├── test_model_structure.py
│   ├── test_forward_backward.py
│   └── test_official_equivalence.py
├── scripts/
│   ├── inspect_model.py
│   └── compare_official.py
├── requirements.txt
└── README.md
```

---

## 4. 实现原则

### 4.1 使用 PyTorch 基础组件

允许使用：

- `torch.Tensor`；
- `torch.nn.Module`；
- `nn.Linear`；
- `nn.Conv2d`；
- `nn.LayerNorm`；
- `nn.GELU`；
- `nn.Dropout`；
- `nn.PixelShuffle`；
- `torch.roll`；
- `torch.meshgrid`；
- `torch.utils.checkpoint`；
- `torch.nn.init.trunc_normal_`。

不使用现成的：

- Swin Transformer 模型；
- SwinIR 模型；
- 第三方 WindowAttention；
- 第三方 RSTB。

### 4.2 尽量保持官方命名

类名和主要成员名建议保持与官方一致：

- `Mlp`；
- `WindowAttention`；
- `SwinTransformerBlock`；
- `BasicLayer`；
- `RSTB`；
- `PatchEmbed`；
- `PatchUnEmbed`；
- `Upsample`；
- `UpsampleOneStep`；
- `SwinIR`。

这样便于：

- 比较 `state_dict`；
- 迁移官方权重；
- 做逐层数值等价验证；
- 阅读官方代码与论文。

### 4.3 禁止只做“外观相似”的简化

以下细节必须实现：

- 相对位置偏置；
- 相对位置索引；
- attention mask；
- W-MSA/SW-MSA 交替；
- cyclic shift 与 reverse shift；
- 两级 LayerNorm 和残差；
- DropPath；
- RSTB 内 token/feature map 转换；
- 主干末尾 LayerNorm；
- 长残差连接；
- 输入尺寸填充与输出裁剪；
- 同尺寸分支的图像级残差。

---

## 5. 基础工具模块

文件：

```text
swinir/common.py
```

### 5.1 `to_2tuple`

功能：

```text
int → (int, int)
tuple → tuple
```

### 5.2 `DropPath`

实现逐样本 stochastic depth。

要求：

- 训练模式下按样本随机丢弃残差分支；
- 推理模式保持恒等；
- 保持期望值不变；
- 支持任意维度张量。

### 5.3 截断正态初始化

优先使用：

```python
torch.nn.init.trunc_normal_
```

初始化规则：

- `nn.Linear.weight`：截断正态，标准差 0.02；
- `nn.Linear.bias`：0；
- `nn.LayerNorm.weight`：1；
- `nn.LayerNorm.bias`：0；
- 卷积层保持 PyTorch 默认初始化，以匹配官方实现。

---

## 6. MLP

文件：

```text
swinir/mlp.py
```

类：

```python
class Mlp(nn.Module)
```

结构：

```text
Linear
→ GELU
→ Dropout
→ Linear
→ Dropout
```

输入输出：

```text
[B, L, C] → [B, L, C]
```

默认隐藏维度：

\[
C_{\mathrm{hidden}}=\mathrm{mlp\_ratio}\times C
\]

---

## 7. 窗口操作

文件：

```text
swinir/window_ops.py
```

### 7.1 `window_partition`

输入：

```text
[B, H, W, C]
```

输出：

```text
[B × nW, M, M, C]
```

要求：

- \(H,W\) 必须能被窗口大小 \(M\) 整除；
- 使用 reshape/permute；
- 不复制不必要数据。

### 7.2 `window_reverse`

输入：

```text
[B × nW, M, M, C]
```

输出：

```text
[B, H, W, C]
```

必须满足：

\[
\operatorname{window\_reverse}
(
\operatorname{window\_partition}(X)
)=X
\]

---

## 8. WindowAttention

文件：

```text
swinir/window_attention.py
```

类：

```python
class WindowAttention(nn.Module)
```

### 8.1 参数

- `dim`；
- `window_size=(M,M)`；
- `num_heads`；
- `qkv_bias`；
- `qk_scale`；
- `attn_drop`；
- `proj_drop`。

### 8.2 相对位置偏置表

形状：

\[
[(2M-1)(2M-1),\ \mathrm{num\_heads}]
\]

### 8.3 相对位置索引

为窗口内任意两个 token 建立二维相对坐标索引：

\[
(M^2,M^2)
\]

要求注册为 buffer，而不是训练参数。

### 8.4 QKV

输入：

```text
[BnW, M², C]
```

通过单个线性层生成 Q、K、V：

```text
[BnW, M², 3C]
```

再重排为多头格式。

### 8.5 注意力

\[
A=
\operatorname{SoftMax}
\left(
\frac{QK^T}{\sqrt d}
+B_{\mathrm{relative}}
+\mathrm{Mask}
\right)
\]

输出：

\[
Y=A V
\]

最后经过：

```text
Linear projection
→ Dropout
```

### 8.6 Mask 支持

- 普通窗口：`mask=None`；
- 移位窗口：加入窗口隔离 mask；
- mask 广播必须覆盖 batch 与 head 维度。

---

## 9. SwinTransformerBlock

文件：

```text
swinir/swin_block.py
```

类：

```python
class SwinTransformerBlock(nn.Module)
```

### 9.1 结构

\[
X_1=X+\operatorname{DropPath}
\left[
\operatorname{MSA}(\operatorname{LN}(X))
\right]
\]

\[
X_2=X_1+\operatorname{DropPath}
\left[
\operatorname{MLP}(\operatorname{LN}(X_1))
\right]
\]

### 9.2 前向过程

1. 保存 shortcut；
2. LayerNorm；
3. `[B,H×W,C] → [B,H,W,C]`；
4. 若 `shift_size>0`，执行负方向 cyclic shift；
5. 划分窗口；
6. 窗口注意力；
7. 恢复窗口；
8. 执行 reverse cyclic shift；
9. 恢复 token 序列；
10. 第一条残差；
11. LayerNorm + MLP；
12. 第二条残差。

### 9.3 shift 规则

- 偶数序号 Block：`shift_size=0`；
- 奇数序号 Block：`shift_size=window_size//2`。

### 9.4 Attention Mask

根据当前 `x_size` 生成区域标签，并通过窗口内标签差构造：

```text
同一区域 → 0
不同区域 → 大负数
```

必须支持动态输入尺寸。

---

## 10. BasicLayer

文件：

```text
swinir/basic_layer.py
```

类：

```python
class BasicLayer(nn.Module)
```

职责：

- 创建指定数量的 `SwinTransformerBlock`；
- 交替设置 shift；
- 按顺序执行 Block；
- 可选 gradient checkpoint；
- 当前 SwinIR 路径不执行空间下采样。

输入输出：

```text
[B,H×W,C] → [B,H×W,C]
```

---

## 11. PatchEmbed 与 PatchUnEmbed

文件：

```text
swinir/patch_ops.py
```

### 11.1 PatchEmbed

当前 `patch_size=1`，不进行卷积下采样。

执行：

```text
[B,C,H,W]
→ flatten spatial
→ transpose
→ [B,H×W,C]
```

主干入口可选 LayerNorm。

### 11.2 PatchUnEmbed

执行逆变换：

```text
[B,H×W,C]
→ transpose/reshape
→ [B,C,H,W]
```

### 11.3 注意

模型 `patch_size=1` 与未来训练裁剪尺寸是完全不同的参数。

---

## 12. RSTB

文件：

```text
swinir/rstb.py
```

类：

```python
class RSTB(nn.Module)
```

### 12.1 主体

```text
BasicLayer
→ PatchUnEmbed
→ 局部卷积
→ PatchEmbed
→ 与 RSTB 输入相加
```

公式：

\[
T_{\mathrm{out}}
=
T_{\mathrm{in}}
+
\operatorname{PatchEmbed}
\left[
\operatorname{Conv}
\left(
\operatorname{PatchUnEmbed}
(
\operatorname{BasicLayer}(T_{\mathrm{in}})
)
\right)
\right]
\]

### 12.2 `1conv` 模式

单个：

```text
Conv2d(C,C,3,1,1)
```

当前第一版使用该模式。

### 12.3 `3conv` 模式

为了完整兼容官方结构，可实现：

```text
3×3 Conv: C → C/4
→ LeakyReLU
→ 1×1 Conv: C/4 → C/4
→ LeakyReLU
→ 3×3 Conv: C/4 → C
```

该模式当前不启用，但建议实现以保持接口完整。

---

## 13. 上采样模块

文件：

```text
swinir/upsample.py
```

### 13.1 `Upsample`

支持：

- \(2^n\) 倍 PixelShuffle；
- 3 倍 PixelShuffle。

### 13.2 `UpsampleOneStep`

单卷积生成 \(s^2C_{\mathrm{out}}\) 通道，再 PixelShuffle。

### 13.3 当前任务要求

SAR 同尺寸重聚焦使用：

```python
upsampler=""
upscale=1
```

因此上采样模块不参与当前前向过程。

实现优先级：

- 同尺寸分支：必须；
- 上采样模块：为完整 SwinIR 兼容性实现，可排在同尺寸分支之后。

---

## 14. SwinIR 主网络

文件：

```text
swinir/model.py
```

类：

```python
class SwinIR(nn.Module)
```

### 14.1 浅层特征

```text
input
→ mean/img_range normalization
→ conv_first
```

### 14.2 深层特征

```text
conv_first output
→ PatchEmbed
→ optional absolute position embedding
→ pos_drop
→ 多个 RSTB
→ final LayerNorm
→ PatchUnEmbed
→ conv_after_body
→ 与浅层特征相加
```

### 14.3 同尺寸恢复分支

```text
fused feature
→ conv_last
→ 与归一化输入相加
→ inverse normalization
→ 裁剪回原始尺寸
```

### 14.4 输入尺寸处理

实现 `check_image_size`：

- 将 H/W 反射填充到 `window_size` 的整数倍；
- 前向结束后裁剪回原始尺寸；
- 测试过小尺寸时应避免非法反射填充。

### 14.5 输出通道

保持官方行为：

```text
num_out_ch = in_chans
```

第一版预设：

```python
in_chans=2
```

因此：

```text
[B,2,H,W] → [B,2,H,W]
```

---

## 15. 第一版配置

```yaml
model:
  img_size: 64
  patch_size: 1
  in_chans: 2
  embed_dim: 180
  depths: [6, 6, 6, 6, 6, 6]
  num_heads: [6, 6, 6, 6, 6, 6]
  window_size: 8
  mlp_ratio: 2.0
  qkv_bias: true
  qk_scale: null
  drop_rate: 0.0
  attn_drop_rate: 0.0
  drop_path_rate: 0.1
  ape: false
  patch_norm: true
  use_checkpoint: false
  upscale: 1
  img_range: 1.0
  upsampler: ""
  resi_connection: "1conv"
```

当前 `in_chans=2` 只是未来 SAR 实部/虚部的预设，不代表输入形式已经最终确定。

---

## 16. 测试规范

### 16.1 单元测试

必须完成：

- `window_partition/window_reverse` 可逆；
- relative position index 形状正确；
- Attention 输出形状正确；
- mask 后无跨窗口错误连接；
- W-MSA Block 前向；
- SW-MSA Block 前向；
- BasicLayer 前向；
- PatchEmbed/UnEmbed 可逆；
- RSTB 输入输出形状一致；
- 完整模型输入输出同尺寸；
- 前向无 NaN/Inf；
- 反向梯度存在且有限。

### 16.2 结构测试

核对：

- 6 个 RSTB；
- 每个 RSTB 6 个 Block；
- W-MSA/SW-MSA 交替；
- `conv_first` 输入通道为 2；
- `conv_last` 输出通道为 2；
- final LayerNorm 存在；
- `conv_after_body` 存在；
- `patch_size=1`；
- `upscale=1`；
- `upsampler=""`。

### 16.3 参数量测试

自实现模型与官方同配置模型参数量必须一致。

允许的差异：

- 仅当明确省略了不参与当前分支的模块时；
- 差异必须记录并解释。

### 16.4 数值等价测试

开发环境中临时加载官方参考实现：

1. 使用相同小配置实例化官方模型与自实现模型；
2. 将官方 `state_dict` 加载到自实现模型；
3. 设置 `eval()`；
4. 使用同一随机输入；
5. 比较输出。

建议标准：

```text
max_abs_error < 1e-6（FP32，CPU）
```

若命名不完全一致，可编写显式权重映射。

参考实现仅用于测试，不作为项目运行依赖。

---

## 17. 许可与来源

如果代码直接参考、翻译或改写官方实现，应：

- 保留 Apache-2.0 许可文件；
- 在源码头部注明论文与官方仓库；
- 标明哪些部分为独立重构；
- 不删除原始版权与许可要求。

---

## 18. 完成定义

- [ ] 所有核心类均由本项目自行实现；
- [ ] 运行时不导入官方 SwinIR；
- [ ] 同尺寸前向分支完整；
- [ ] 单元测试全部通过；
- [ ] 结构与参数量匹配；
- [ ] 官方权重可以通过直接或映射方式加载；
- [ ] 数值等价测试通过；
- [ ] 双通道随机输入前向/反向通过；
- [ ] 参考版本、论文与许可证已记录。
