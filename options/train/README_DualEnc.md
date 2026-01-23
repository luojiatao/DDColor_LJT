# 配置文件说明文档：Multi-Scale Detail Refinement (Dual Encoder)

本配置文件 (`ljt_train_detail_refine_dualenc.yml`) 用于训练基于 **Dual Encoder (双编码器)** 策略的衣物褶皱去除模型。
该方案通过两个独立的 Encoder 分别处理“结构图”和“参考图”，并在 Decoder 阶段通过 Cross-Attention 注入纹理，能更好地物理隔离“褶皱”信息。

---

## 1. 基础全局设置 (General Settings)

| 参数名 | 说明 | 示例值 |
| :--- | :--- | :--- |
| `name` | 实验名称 (Output 文件夹名) | `detail_refine_dualenc` |
| `model_type` | 使用的模型封装类 | `DetailRefineModel` |
| `num_gpu` | GPU 数量 | `auto` |
| `manual_seed` | 随机种子 | `0` |

---

## 2. 数据集设置 (Datasets)

需要在 `meta_info_file` 中指定包含 `[B路径 A路径 GT路径]` 的文本文件。

### 关键配置项
| 参数名 | 说明 | 推荐设定 |
| :--- | :--- | :--- |
| `gt_size` | **训练尺寸控制**。`~` (None) 为原图训练；数字为随机裁剪尺寸 (Random Crop)，对 B/A/GT 做一致随机裁剪 | `512` (推荐) |
| `use_hflip` | 随机水平翻转 | `True` |
| `batch_size_per_gpu` | 单卡 Batch Size | `2` |

> **注意**：由于 Dual Encoder 有两个骨干网络，显存占用较高。如果 OOM (显存不足)，请减小 `batch_size_per_gpu` 或开启 `gt_size` 进行裁剪训练。

---

## 3. 网络结构 (Network - Dual Encoder 特有)

对应 `basicsr/archs/ljt_detail_refine_dualenc_arch.py`。

### 核心差异参数
| 参数名 | 说明 | 关键作用 |
| :--- | :--- | :--- |
| `type` | 网络架构类名 | `DetailRefineDualEnc` |
| `encoder_name_b` | **结构分支** (Branch B) 的骨干网络。处理模糊但无褶皱的图。 | 决定生成图的**空间结构**和轮廓 |
| `encoder_name_a` | **参考分支** (Branch A) 的骨干网络。处理清晰但有褶皱的图。 | 提供**高频纹理**信息 |
| `highpass_kernel` | **高通滤波器尺寸**。在 A 分支输入前执行 `Image - LowPass`。 | **关键防褶皱参数**。值越大，滤除的低频成分(阴影/褶皱)越多，只保留纹理。建议 `5` 或 `7` |

### 通用架构参数
| 参数名 | 说明 |
| :--- | :--- |
| `num_queries` | 纹理 Token 数量 |
| `input_size` | 初始化构建网络用的 Dummy 输入尺寸（只用于 __init__ 里 dummy forward，**不会** resize/crop 真实训练输入） |
| `num_scales` | 多尺度层级 (固定为3) |
| `nf` | 内部特征通道数 (512) |
| `encoder_from_pretrain` | 两个 Encoder 是否都加载 ImageNet 权重 |

---

## 4. 路径设置 (Path)

| 参数名 | 说明 |
| :--- | :--- |
| `pretrain_network_g` | 预训练/微调权重路径 |
| `strict_load_g` | 是否严格加载 |
| `resume_state` | 恢复训练状态路径 |

---

## 5. 训练参数 (Training)

| 参数名 | 说明 |
| :--- | :--- |
| `optim_g` | 优化器配置 |
| `scheduler` | 学习率调度 |
| `total_iter` | 总迭代数 (e.g. 300000) |
| `pixel_opt` | 像素损失权重 (L1 Loss) |
| `perceptual_opt` | 感知损失配置 (VGG19)。对于恢复衣物纹理至关重要。 |

---

## 6. 常见问题 (FAQ)

### Q: 为什么显存占用比 EarlyFusion 大很多？
**A**: EarlyFusion 只有一个 Encoder，而 DualEnc 有两个完整的 ConvNeXt-Large Encoder 同时运行。

### Q: `highpass_kernel` 设为 0 或 1 会怎样？
**A**: 会关闭高通滤波，参考图 A 会原样进入 Encoder A。这会导致模型可能直接拷贝 A 图中的褶皱阴影，失去“去褶皱”的效果。

### Q: 如何完全禁用数据增强？
**A**: 将 `gt_size` 设为 `~`，`use_hflip` 设为 `False`，`use_rot` 设为 `False`。
