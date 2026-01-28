#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# 方案DDColorMask复现：Early-Fusion + refine 同时拼 B 与 A
#
# 目标：用 A(清晰但有褶皱) + B(无褶皱但模糊) 生成 C(无褶皱且清晰)。
#
# 对应代码入口：
# - 网络：basicsr/archs/ljt_detail_refine_ddcolormask_arch.py   -> DetailRefineDDColorMask
# - 数据集：basicsr/data/ljt_triplet_refine_dataset.py          -> TripletRefineDataset
# - 训练模型：basicsr/models/ljt_detail_refine_model.py         -> DetailRefineModel
# - 训练配置：options/train/ljt_train_detail_refine_ddcolormask.yml
#
# ---------------------- 网络结构（通俗说明） ----------------------
# 输入：B(3ch) 与 A(3ch)
# 1) Early-Fusion：通道拼接成 6ch，送入同一个 ConvNeXt 编码器：
#        x = concat([B, A])  ->  Encoder(ConvNeXt, in_chans=6)
# 2) Decoder：复用 DDColor 的“像素分支(U-Net上采样) + token分支(queries+attn)”结构，输出 Q 个特征图 out_feat。
# 3) Refine（本方案关键点）：末端同时拼 B 与 A：
#        concat([out_feat(Qch), B(3ch), A(3ch)]) -> 1x1 conv -> 输出 RGB
#
# ---------------------- 数据与训练要求 ----------------------
# meta 文件每行三列：
#   path_to_B  path_to_A  path_to_C(GT)
# 且三张图必须同尺寸；训练会对三者做一致 crop/flip/rotate。
#
# 运行前请确保：
# 1) 已准备好 data_list/ljt_detail_refine_train.txt 和 data_list/ljt_detail_refine_val.txt
# 2) 已准备好预训练 ConvNeXt 权重（pretrain/convnext_large_22k_224.pth）
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

# 生成带时间戳的实验名称，格式：detail_refine_ddcolormask_YYYYMMDD_HHMMSS
TIMESTAMP=$(date +"%m%d%H")
EXP_NAME="detail_refine_ddcolormask_${TIMESTAMP}"
echo "[train] 实验名称: ${EXP_NAME}"

python basicsr/train.py \
  -opt options/train/ljt_train_detail_refine_ddcolormask.yml \
  --launcher none \
  --force_yml name="${EXP_NAME}"
