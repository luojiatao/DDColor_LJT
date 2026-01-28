# 细节修复网络

基于 [DDColor](https://arxiv.org/abs/2212.11613) 架构扩展的图像细节修复方案。

## 输入输出定义

| 符号 | 说明 |
|------|------|
| **A** | 参考图（提供细节/纹理） |
| **B** | 结构图（提供空间布局） |
| **C** | 目标输出 |

## 安装

```sh
conda create -n ddcolor python=3.9
conda activate ddcolor
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install -r requirements.train.txt
python3 setup.py develop
```

## 可用方案

| 方案 | 编码器 | 解码器 | 参数量 | 特点 |
|------|--------|--------|--------|------|
| **ResUnet** | ResNet34 | U-Net | ~25M | 轻量快速，支持残差+Mask 可视化 |
| **DDColorMask** | ConvNeXt-L | U-Net+Transformer | ~200M+ | 大容量，效果上限高 |

## 快速开始

### 1. 准备数据列表

每行三列 `B路径 A路径 GT路径`：
```
/path/to/image_b.jpg /path/to/image_a.jpg /path/to/image_gt.jpg
```

### 2. 训练

**ResUnet 方案**（推荐）：
```sh
bash scripts/ljt_train_detail_refine_resunet.sh
```
- 配置：`options/train/ljt_train_detail_refine_resunet.yml`
- 支持 `residual_and_mask` 输出模式，验证时可视化残差和 Mask

**DDColorMask 方案**：
```sh
bash scripts/ljt_train_detail_refine_ddcolormask.sh
```
- 配置：`options/train/ljt_train_detail_refine_ddcolormask.yml`
- 需要预下载 [ConvNeXt 权重](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth) 到 `pretrain/`

### 3. 数据增强配置

训练配置支持随机模糊增强：
```yaml
use_blur_aug: true                # 总开关
blur_aug_probs: [1, 1, 1]         # 权重 [无处理, 下采样模糊, 高斯模糊]
blur_aug_target: 'b'              # 目标：'b'/'a'/'both'
downsample_scale: 2.0             # 下采样倍数
gaussian_radius: 2                # 高斯核半径
```

## 代码结构

```
basicsr/
├── archs/
│   ├── ljt_detail_refine_resunet_arch.py      # ResUnet 网络
│   └── ljt_detail_refine_ddcolormask_arch.py  # DDColorMask 网络
├── models/
│   ├── ljt_detail_refine_model.py             # 基础训练模型
│   └── ljt_detail_refine_resunet_model.py     # ResUnet 模型（支持可视化）
├── data/
│   └── ljt_triplet_refine_dataset.py          # 三元组数据集
cjy_res_unet/                                   # ResUnet6C 核心实现
options/train/
├── ljt_train_detail_refine_resunet.yml        # ResUnet 配置
└── ljt_train_detail_refine_ddcolormask.yml    # DDColorMask 配置
scripts/
├── ljt_train_detail_refine_resunet.sh         # ResUnet 启动脚本
└── ljt_train_detail_refine_ddcolormask.sh     # DDColorMask 启动脚本
```

## 致谢

- [DDColor](https://github.com/piddnad/DDColor) - 原始架构
- [BasicSR](https://github.com/xinntao/BasicSR) - 训练框架
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) - 编码器骨干
