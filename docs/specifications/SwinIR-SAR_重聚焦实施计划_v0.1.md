# SwinIR 独立架构复现实施计划（Implementation Plan）

> 版本：v0.4  
> 当前任务：使用 PyTorch 基础组件从零实现 SwinIR  
> 当前不需要：真实 SAR 输入、数据集和正式训练  
> 验证基准：论文结构、官方源码、官方同配置模型数值输出

---

## 1. 总体路线

```text
保存官方参考文件
→ 搭建独立项目
→ 实现基础工具
→ 实现窗口操作
→ 实现窗口注意力
→ 实现 Swin Transformer Block
→ 实现 BasicLayer
→ 实现 PatchEmbed/UnEmbed
→ 实现 RSTB
→ 实现 SwinIR 同尺寸主网络
→ 单元测试
→ 参数量对比
→ 官方数值等价测试
```

---

## 2. 参考文件准备

无需 clone 完整仓库。

至少保存：

```text
references/network_swinir.py
references/main_test_swinir.py
references/LICENSE
references/UPSTREAM_COMMIT.txt
```

其中：

- `network_swinir.py`：结构与实现参考；
- `main_test_swinir.py`：官方模型配置参考；
- `LICENSE`：许可与来源记录；
- `UPSTREAM_COMMIT.txt`：固定参考版本。

若不方便单独下载，也可以 shallow clone 后仅将其作为参考：

```bash
git clone --depth 1 https://github.com/JingyunLiang/SwinIR.git references/SwinIR
git -C references/SwinIR rev-parse HEAD > references/UPSTREAM_COMMIT.txt
```

自实现代码不得导入参考模型。

---

## 3. P0：项目初始化

创建目录：

```bash
mkdir -p swinir configs tests scripts references
touch swinir/__init__.py
touch swinir/common.py
touch swinir/mlp.py
touch swinir/window_ops.py
touch swinir/window_attention.py
touch swinir/swin_block.py
touch swinir/basic_layer.py
touch swinir/patch_ops.py
touch swinir/rstb.py
touch swinir/upsample.py
touch swinir/model.py
```

安装：

```bash
pip install torch pytest pyyaml
```

`timm` 不再是运行时必需依赖；DropPath、tuple 转换和初始化由本项目实现或使用 PyTorch 原生接口。

---

## 4. P1：基础工具与 MLP

### Task 1：`to_2tuple`

完成 int/tuple 转换。

### Task 2：`DropPath`

测试：

- eval 模式恒等；
- drop_prob=0 恒等；
- train 模式形状不变；
- 无 NaN/Inf。

### Task 3：初始化函数

只对 Linear 与 LayerNorm 应用官方规则。

### Task 4：`Mlp`

测试：

```text
[B,L,C] → [B,L,C]
```

验收：

- 前向通过；
- 反向通过；
- 参数量正确。

---

## 5. P2：窗口操作

实现：

- `window_partition`；
- `window_reverse`。

测试尺寸：

```text
B=2, H=16, W=24, C=8, window=8
```

验收：

\[
\operatorname{reverse}(\operatorname{partition}(X))=X
\]

并检查非整除尺寸应明确报错或由模型上层先 padding。

---

## 6. P3：WindowAttention

实施顺序：

1. 相对位置偏置表；
2. 相对位置坐标；
3. relative position index；
4. QKV 投影；
5. 多头重排；
6. scaled dot-product；
7. relative bias；
8. mask；
9. softmax；
10. value 聚合；
11. 输出投影。

测试：

- 无 mask；
- 有 mask；
- 不同 batch window 数；
- 输出形状；
- 梯度；
- relative index 范围。

验收：

```text
input : [BnW, M², C]
output: [BnW, M², C]
```

---

## 7. P4：SwinTransformerBlock

实施：

1. LayerNorm；
2. reshape 为二维；
3. cyclic shift；
4. window partition；
5. WindowAttention；
6. window reverse；
7. reverse shift；
8. 第一条残差；
9. LayerNorm + MLP；
10. 第二条残差。

分别测试：

- `shift_size=0`；
- `shift_size=M/2`；
- 动态 `x_size`；
- mask 缓存或重新计算；
- 前向/反向。

验收：

```text
[B,H×W,C] → [B,H×W,C]
```

---

## 8. P5：BasicLayer

实现：

- 深度参数；
- Block 列表；
- shift 交替；
- DropPath 列表；
- 可选 checkpoint。

第一版测试：

```text
depth=6
shift sequence=[0,4,0,4,0,4]
```

验收：

- Block 数量正确；
- shift 顺序正确；
- 输入输出形状一致。

---

## 9. P6：PatchEmbed 与 PatchUnEmbed

实现：

```text
[B,C,H,W] ↔ [B,H×W,C]
```

测试：

- norm 关闭；
- norm 开启；
- 可逆性；
- 动态 H/W。

特别记录：

```text
model patch_size = 1
```

该参数不是训练裁剪大小。

---

## 10. P7：RSTB

实现：

```text
BasicLayer
→ PatchUnEmbed
→ Conv
→ PatchEmbed
→ Residual Add
```

优先完成：

```text
resi_connection="1conv"
```

随后实现可选：

```text
resi_connection="3conv"
```

测试：

- 输入输出形状；
- 残差路径；
- 1conv 参数量；
- 3conv 参数量；
- 梯度。

---

## 11. P8：上采样模块

为了完整兼容官方代码，实现：

- `Upsample`；
- `UpsampleOneStep`。

但当前 SAR 同尺寸分支不依赖它们。

执行顺序上可放在完整同尺寸模型通过之后。

---

## 12. P9：SwinIR 主模型

### 第一阶段先实现同尺寸分支

参数：

```yaml
upscale: 1
upsampler: ""
in_chans: 2
```

实现顺序：

1. mean/img_range；
2. `conv_first`；
3. `PatchEmbed`；
4. absolute position embedding 可选；
5. `pos_drop`；
6. 多个 RSTB；
7. final LayerNorm；
8. `PatchUnEmbed`；
9. `conv_after_body`；
10. 长残差；
11. `conv_last`；
12. 与输入相加；
13. 逆归一化；
14. 裁剪。

随后再补充：

- `pixelshuffle`；
- `pixelshuffledirect`；
- `nearest+conv`。

---

## 13. P10：完整模型测试

使用较小配置先调试：

```yaml
img_size: 16
window_size: 4
embed_dim: 24
depths: [2, 2]
num_heads: [3, 3]
in_chans: 2
upscale: 1
upsampler: ""
```

通过后再切换标准配置：

```yaml
img_size: 64
window_size: 8
embed_dim: 180
depths: [6,6,6,6,6,6]
num_heads: [6,6,6,6,6,6]
```

测试：

```text
input : [1,2,64,64]
output: [1,2,64,64]
```

并执行反向传播。

---

## 14. P11：与官方实现对照

### 14.1 结构对照

比较：

- 模块数量；
- 每层参数形状；
- `state_dict` key；
- 总参数量；
- 可训练参数量。

### 14.2 数值等价

流程：

1. 创建相同小配置；
2. 官方与自实现模型均设为 eval；
3. 将官方权重加载到自实现模型；
4. 输入同一随机张量；
5. 比较中间层和最终输出。

推荐逐级比较：

```text
WindowAttention
→ SwinTransformerBlock
→ RSTB
→ 完整 SwinIR
```

阈值：

```text
FP32 CPU max_abs_error < 1e-6
```

若不一致，按以下顺序排查：

1. reshape/permute 顺序；
2. relative position index；
3. attention mask；
4. shift 方向；
5. DropPath/eval 状态；
6. LayerNorm 位置；
7. RSTB token/feature 转换；
8. mean/img_range；
9. padding/cropping；
10. 初始化与权重映射。

---

## 15. 开发顺序与预计工作单元

### Milestone 1：注意力基础

- common；
- MLP；
- window ops；
- WindowAttention。

完成后可独立测试局部窗口注意力。

### Milestone 2：Swin 主体

- SwinTransformerBlock；
- BasicLayer；
- PatchEmbed/UnEmbed。

完成后可测试多层 W-MSA/SW-MSA。

### Milestone 3：SwinIR 主干

- RSTB；
- conv_after_body；
- 长残差；
- 同尺寸重建分支。

完成后获得可运行的第一版架构。

### Milestone 4：官方对照

- 参数量一致；
- 权重可映射；
- 数值等价；
- 文档归档。

### Milestone 5：完整分支兼容

- Upsample；
- UpsampleOneStep；
- PixelShuffle 分支；
- nearest+conv 分支。

该里程碑对当前 SAR 同尺寸任务不是阻塞项。

---

## 16. 当前不实施的内容

- SAR 输入构造；
- 幅度/相位编码；
- Dataset；
- Charbonnier loss；
- optimizer；
- scheduler；
- 正式训练；
- PFA/BP 配对；
- refocusing 指标；
- 泛化。

---

## 17. 完成标准

- [ ] 运行时不依赖官方 SwinIR；
- [ ] 所有核心模块均自主实现；
- [ ] 同尺寸分支可前向和反向；
- [ ] 标准 6×6 配置可实例化；
- [ ] 双通道输入输出同尺寸；
- [ ] 单元测试通过；
- [ ] 参数量与官方一致；
- [ ] 官方权重可加载或映射；
- [ ] 逐层及最终数值等价通过；
- [ ] 来源、commit 与许可证已记录。
