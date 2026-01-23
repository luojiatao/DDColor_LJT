# 配置文件说明文档：Multi-Scale Detail Refinement (DDColorMask 复现)

本配置文件 (`ljt_train_detail_refine_ddcolormask.yml`) 用于训练基于 **DDColorMask 风格的 Early-Fusion** 细节修复/去褶皱模型。

核心思路：
- 输入两张图：B（结构/内容图：无褶皱但可能模糊）与 A（参考/细节图：清晰但可能有褶皱）。
- 先在输入端把 B/A 做通道拼接（Concat）后走单个 Encoder + 多尺度 Decoder。
- **与 EarlyFusion 基线的关键区别**：DDColorMask 复现版本在 refine 阶段会 **同时拼回 B 与 A**（即末端 late-fusion），更接近你参考的 `ddcolormask_arch_cjy_detail_transfer.py` 设计。

---

## 1. 基础全局设置 (General Settings)

| 参数名 | 说明 | 示例值 |
| :--- | :--- | :--- |
| `name` | 实验名称，决定日志与模型保存目录 | `detail_refine_ddcolormask` |
| `model_type` | 使用的模型封装类（训练/验证逻辑） | `DetailRefineModel` |
| `scale` | 放大倍率（修复类任务通常为 1） | `1` |
| `num_gpu` | GPU 数量，`auto` 为自动检测 | `auto` |
| `manual_seed` | 随机种子 | `0` |

---

## 2. 数据集设置 (Datasets)

数据加载基于 `basicsr/data/ljt_triplet_refine_dataset.py`，需要准备一个 meta 文件，每行三列路径：

```
path_to_B   path_to_A   path_to_C
```

其中：
- B：结构/内容图（无褶皱但模糊）
- A：参考/细节图（清晰但有褶皱）
- C：GT 真值（无褶皱且清晰）

### 训练集 (train)

| 参数名 | 说明 | 推荐设定 |
| :--- | :--- | :--- |
| `type` | 数据集类名 | `TripletRefineDataset` |
| `meta_info_file` | 数据列表路径 | `data_list/ljt_detail_refine_train.txt` |
| `gt_size` | **训练裁剪尺寸**。数字表示随机裁剪 patch 大小，会对 B/A/GT 做一致随机裁剪 | `512` (推荐) |
| `use_hflip` | 随机水平翻转 | `True` |
| `use_rot` | 随机旋转（90/180/270） | `False` |
| `batch_size_per_gpu` | 单卡 batch size | `2` |

### 验证集 (val)

| 参数名 | 说明 | 推荐设定 |
| :--- | :--- | :--- |
| `meta_info_file` | 验证列表路径 | `data_list/ljt_detail_refine_val.txt` |
| `gt_size` | 验证是否裁剪。`~` 表示不裁剪、保持原图验证 | `~` |

> 注意：如果你的原图任意边长小于 `gt_size`，训练会报错（因为随机裁剪无法进行）。

---

## 3. 网络结构 (Network)

对应 `basicsr/archs/ljt_detail_refine_ddcolormask_arch.py`。

| 参数名 | 说明 | 当前配置 |
| :--- | :--- | :--- |
| `type` | 网络架构类名 | `DetailRefineDDColorMask` |
| `encoder_name` | 骨干网络（ConvNeXt） | `convnext-l` |
| `num_queries` | Transformer Decoder 的 token 数量（容量/细节表达） | `256` |
| `num_scales` | 多尺度层级数 | `3` |
| `dec_layers` | Transformer Decoder 层数 | `9` |
| `nf` | 内部特征通道数（计算量相关） | `512` |
| `out_channels` | 输出通道数 | `3` |
| `last_norm` | 解码器末端归一化 | `Spectral` |
| `encoder_from_pretrain` | 是否加载 ImageNet 预训练初始化 | `True` |
| `input_size` | 初始化构建网络用的 Dummy 输入尺寸（只用于 `__init__` 里 dummy forward，**不会** resize/crop 真实训练输入） | `[512, 512]` |
| `do_imagenet_normalize` | 网络内部是否做 ImageNet mean/std 归一化 | `True` |

### 关于 `input_size` 的常见误解

- `input_size` **不是**“把训练数据 resize 到该尺寸”。
- 真实训练输入的尺寸由 dataset 产出决定：
  - 你把 `gt_size` 设为 `512`，训练时就是随机裁剪 512×512。
  - 你把 `gt_size` 设为 `~`，训练时就是原图尺寸（可能显存更吃紧）。

---

## 4. 路径设置 (Path)

| 参数名 | 说明 |
| :--- | :--- |
| `pretrain_network_g` | 预训练/微调权重路径（`.pth`） |
| `strict_load_g` | 是否严格加载权重键值 |
| `resume_state` | 恢复训练状态路径（`.state`，含优化器与调度器） |

---

## 5. 训练参数 (Training)

| 参数名 | 说明 |
| :--- | :--- |
| `optim_g` | 优化器（AdamW）与学习率/权重衰减 |
| `scheduler` | 学习率调度（MultiStepLR） |
| `total_iter` | 总迭代数 |
| `pixel_opt` | 像素损失（L1） |
| `perceptual_opt` | 感知损失（VGG19），对纹理恢复重要 |

---

## 6. 运行方式 (How to Run)

### 直接启动

```bash
python basicsr/train.py -opt options/train/ljt_train_detail_refine_ddcolormask.yml --launcher none
```

### 使用脚本启动

仓库内也提供了脚本（便于与你的其它方案脚本风格保持一致）：

```bash
bash scripts/ljt_train_detail_refine_ddcolormask.sh
```
