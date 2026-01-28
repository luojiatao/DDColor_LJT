#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# 方案 ResUnet：轻量级 ResNet34 + U-Net 细节修复
#
# 目标：用 A(清晰但有褶皱) + B(无褶皱但模糊) 生成 C(无褶皱且清晰)。
#
# 对应代码入口：
# - 网络：basicsr/archs/ljt_detail_refine_resunet_arch.py     -> DetailRefineResUnet
# - 数据集：basicsr/data/ljt_triplet_refine_dataset.py        -> TripletRefineDataset
# - 训练模型：basicsr/models/ljt_detail_refine_resunet_model.py -> DetailRefineResUnetModel（支持可视化残差/Mask）
# - 训练配置：options/train/ljt_train_detail_refine_resunet.yml
#
# ---------------------- 网络结构（通俗说明） ----------------------
# 输入：B(3ch) 与 A(3ch)
# 1) Early-Fusion：通道拼接成 6ch，送入 ResNet34 编码器
# 2) U-Net 解码器：跳跃连接 + 上采样
# 3) 输出模式：
#    - rgb: 直接输出 RGB
#    - residual: 学习残差，output = residual + B
#    - residual_and_mask: 学习残差和 mask，output = residual * mask + B
#
# ---------------------- 与 DDColorMask 方案对比 ----------------------
# | 特性        | ResUnet      | DDColorMask      |
# |------------|--------------|------------------|
# | 编码器      | ResNet34     | ConvNeXt-L       |
# | 解码器      | U-Net        | U-Net+Transformer|
# | 参数量      | ~25M         | ~200M+           |
# | batch_size  | 24           | 12               |
# | 训练速度    | 快           | 慢               |
#
# ---------------------- 数据与训练要求 ----------------------
# meta 文件每行三列：
#   path_to_B  path_to_A  path_to_C(GT)
# 且三张图必须同尺寸；训练会对三者做一致 crop/flip/rotate。
#
# 运行前请确保：
# 1) 已准备好 data_list/ljt_detail_refine_train.txt 和 data_list/ljt_detail_refine_val.txt
# 2) ResNet34 预训练权重会自动从 torchvision 下载
# ==========================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACTIVATE_HELPER="$ROOT_DIR/scripts/_activate_conda.sh"
if [[ ! -f "$ACTIVATE_HELPER" ]]; then
  echo "[train] missing conda helper: $ACTIVATE_HELPER" >&2
  exit 1
fi
source "$ACTIVATE_HELPER" ddcolor

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# 生成带时间戳的实验名称
TIMESTAMP=$(date +"%m%d%H")
EXP_NAME="detail_refine_resunet_${TIMESTAMP}"
echo "[train] 实验名称: ${EXP_NAME}"

python basicsr/train.py \
  -opt options/train/ljt_train_detail_refine_resunet.yml \
  --launcher none \
  --force_yml name="${EXP_NAME}"
