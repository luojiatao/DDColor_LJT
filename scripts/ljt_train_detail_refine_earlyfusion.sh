#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# 方案1：单编码器 Early-Fusion（基线）
#
# 目标：用 A(清晰但有褶皱) 提供细节，用 B(无褶皱但模糊) 提供结构，生成 C(无褶皱且清晰)。
# 关键约束：禁止“把 A 的褶皱带回输出”。
#
# 对应代码入口：
# - 网络：basicsr/archs/ljt_detail_refine_earlyfusion_arch.py  -> DetailRefineEarlyFusion
# - 数据集：basicsr/data/ljt_triplet_refine_dataset.py         -> TripletRefineDataset
# - 训练模型：basicsr/models/ljt_detail_refine_model.py        -> DetailRefineModel
# - 训练配置：options/train/ljt_train_detail_refine_earlyfusion.yml
#
# ---------------------- 网络结构（严谨说明） ----------------------
# 输入：B(3ch) 与 A(3ch)
# 1) Early-Fusion：在通道维拼接成 6ch，送入同一个 ConvNeXt 编码器：
#        x = concat([B, A])  ->  Encoder(ConvNeXt, in_chans=6)
#
# 2) Decoder 沿用 DDColor 的“双分支”思想（注意：这不是“两输入分支”）：
#    - 像素分支：U-Net 式上采样，决定输出的空间网格/对齐能力；
#    - token 分支：learnable queries + cross/self-attn，从多尺度特征提取可注入的“纹理 token”。
#
# 3) Anti-wrinkle 关键点：末端 refine 阶段只拼 B，不拼 A：
#        out_feat(Qch) 与 B(3ch) 拼接 -> 1x1 conv -> 输出 RGB
#
# 为什么这样能降低“抄 A 褶皱”的概率？
# - 如果在 refine 里把 A 再拼回去（late-fusion），网络会学到一条非常短的捷径：
#   直接从 A 把低频明暗/阴影（常对应褶皱）拷贝到输出。
# - 这里强制 refine 只能看 B（结构锚点），A 的信息只能通过 Encoder/Token 的受限通道进入。
#
# ---------------------- 数据与训练要求 ----------------------
# meta 文件每行三列：
#   path_to_B  path_to_A  path_to_C(GT)
# 且三张图必须同尺寸；训练会对三者做一致 crop/flip/rotate。
#
# 运行前请确保：
# 1) 已准备好 data_list/ljt_detail_refine_train.txt 和 data_list/ljt_detail_refine_val.txt
# 2) 已安装依赖并能正常 import basicsr/torch/opencv 等
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

python basicsr/train.py \
  -opt options/train/ljt_train_detail_refine_earlyfusion.yml \
  --launcher none
