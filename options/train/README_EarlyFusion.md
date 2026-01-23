# 配置文件说明文档：Multi-Scale Detail Refinement (Early Fusion)

本配置文件 (`ljt_train_detail_refine_earlyfusion.yml`) 用于训练基于 **Early Fusion (早期融合)** 策略的衣物褶皱去除模型。
该方案通过将结构图（模糊/无褶皱）与参考图（清晰/有褶皱）在输入端拼接（Concat），送入单编码器进行特征提取。

---

## 1. 基础全局设置 (General Settings)

| 参数名 | 说明 | 示例值 |
| :--- | :--- | :--- |
| `name` | 实验名称，决定日志和模型保存的文件夹名称 | `detail_refine_earlyfusion` |
| `model_type` | 使用的模型封装类，对应 `models/ljt_detail_refine_model.py` | `DetailRefineModel` |
| `scale` | 放大倍率（图像修复任务通常为 1） | `1` |
| `num_gpu` | 使用的 GPU 数量，`auto` 为自动检测 | `auto` |
| `manual_seed` | 随机种子，保证实验可复现性 | `0` |

---

## 2. 数据集设置 (Datasets)

数据加载基于 `basicsr/data/ljt_triplet_refine_dataset.py`。需要准备包含三列路径（B A GT）的文本文件。

### 训练集 (train)
| 参数名 | 说明 | 推荐设定 |
| :--- | :--- | :--- |
| `name` | 数据集逻辑名称 | `TripletTrain` |
| `type` | 数据集类名 | `TripletRefineDataset` |
| `meta_info_file` | 数据列表文件路径 (支持列表格式) | `["data_list/train.txt"]` |
| `gt_size` | **训练裁剪尺寸**。`~` (None) 表示使用原图训练；数字表示随机裁剪的 patch 大小（对 B/A/GT 做一致随机裁剪） | `512` (推荐) |
| `use_hflip` | 是否开启随机水平翻转增强 | `True` |
| `use_rot` | 是否开启随机旋转 (90/180/270度) | `False` |
| `use_shuffle` | 每个 epoch 是否打乱数据 | `true` |
| `num_worker_per_gpu` | 如果 CPU 较强建议设高，加速读取 | `8` |
| `batch_size_per_gpu` | 单卡 Batch Size | `2` |
| `dataset_enlarge_ratio` | 虚拟放大 Epoch 长度 (仅影响打印频率，不影响总 Iter) | `1` |

### 验证集 (val)
| 参数名 | 说明 | 推荐设定 |
| :--- | :--- | :--- |
| `meta_info_file` | 验证集列表文件路径 | `"data_list/val.txt"` |
| `val_freq` | (定义在 `val` 节) 多少个 Iter验证一次 | `5000` |

---

## 3. 网络结构 (Network)

对应 `basicsr/archs/ljt_detail_refine_earlyfusion_arch.py`。

| 参数名 | 说明 | 备注 |
| :--- | :--- | :--- |
| `type` | 网络架构类名 | `DetailRefineEarlyFusion` |
| `encoder_name` | 骨干网络类型 (支持 `convnext-t/s/b/l`) | `convnext-l` 容量最大 |
| `num_queries` | Transformer Decoder 中的纹理 Token 数量 | `256` |
| `num_scales` | 多尺度特征层级数 (对应代码架构固定为3) | `3` (固定) |
| `dec_layers` | Transformer Decoder 层数 | `9` |
| `nf` | 内部特征通道数 (影响计算量) | `512` |
| `out_channels` | 输出图片通道数 | `3` (RGB) |
| `last_norm` | 解码器末端的归一化方式 | `Spectral` 或 `Weight` |
| `encoder_from_pretrain` | 是否加载 ImageNet 预训练权重初始化 Encoder | `True` |
| `input_size` | 初始化构建网络用的 Dummy 输入尺寸（只用于 __init__ 里 dummy forward，**不会** resize/crop 真实训练输入） | `[512, 512]` |
| `do_imagenet_normalize` | 是否在网络内部对输入做 ImageNet 均值方差归一化 | `True` |

---

## 4. 路径设置 (Path)

| 参数名 | 说明 |
| :--- | :--- |
| `pretrain_network_g` | **预训练模型路径**。如果是断点续训或微调，在此指定 `.pth` 文件路径 |
| `strict_load_g` | 是否严格匹配参数键值 |
| `resume_state` | **恢复训练状态路径** (`.state` 文件)，用于恢复优化器和 Scheduler 状态 |

---

## 5. 训练参数 (Training)

| 参数名 | 说明 |
| :--- | :--- |
| `optim_g` | 优化器配置 (类型、学习率、权重衰减等) |
| `scheduler` | 学习率调整策略 (MultiStepLR, CosineAnnealing 等) |
| `total_iter` | 总训练迭代次数 (Iterations) |
| `warmup_iter` | 预热迭代次数，`-1` 表示关闭 |
| `pixel_opt` | **像素损失** (L1/L2/Charbonnier)。`loss_weight` 控制权重 |
| `perceptual_opt` | **感知损失** (VGG)。推荐开启以恢复纹理。`layer_weights` 控制不同层权重 |

---

## 6. 日志与验证 (Logger & Validation)

| 参数名 | 说明 |
| :--- | :--- |
| `save_img` | 验证时是否保存可视化结果图片 |
| `metrics` | 验证指标配置 (如 PSNR, SSIM)。`crop_border` 控制是否裁去边缘像素 (SR常用，这里设0) |
| `print_freq` | 终端打印 Loss 的频率 (Iter) |
| `save_checkpoint_freq` | 保存模型权重的频率 (Iter) |
| `use_tb_logger` | 是否使用 TensorBoard 记录曲线 |
