#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# 方案3：双编码器（效果上限优先）
#   B 控结构 + A 高频纹理记忆 + 受控注入（避免 A 褶皱回流）
#
# 目标：用 A(清晰但有褶皱) 提供“细节/纹理”，用 B(无褶皱但模糊) 提供“结构/网格”，输出 C(无褶皱且清晰)。
#
# 对应代码入口：
# - 网络：basicsr/archs/ljt_detail_refine_dualenc_arch.py       -> DetailRefineDualEnc
# - 数据集：basicsr/data/ljt_triplet_refine_dataset.py          -> TripletRefineDataset
# - 训练模型：basicsr/models/ljt_detail_refine_model.py         -> DetailRefineModel
# - 训练配置：options/train/ljt_train_detail_refine_dualenc.yml
#
# ---------------------- 网络结构（严谨说明） ----------------------
# 输入：B(3ch) 与 A(3ch)
#
# 1) 双编码器拆职责：
#    - Encoder_B：只编码 B，用于“空间网格/结构锚点”；
#    - Encoder_A：只编码 A，但先做高通，仅保留更偏高频的纹理信息。
#
# 2) A 高频化（降低褶皱低频阴影的进入概率）：
#        A_hp = A - Blur(A)   （实现用 avg_pool2d 近似 blur，可微、代价低）
#
# 3) 像素分支（只走 B）：
#        img_features_B = PixelDecoder( hooks_from_Encoder_B )
#    这一步非常关键：输出的空间网格由 B 决定，天然对齐“无褶皱结构”。
#
# 4) token 分支（memory 来自 A_hp 的多尺度特征）：
#    - learnable queries 通过 cross-attn 读取 A 的多尺度 memory（更像“纹理记忆库”）；
#    - 受控注入：token 不能直接变成像素图，只能通过与 img_features_B 的相关性投影回空间：
#        out_feat = einsum(token_embed, img_features_B)
#    直观理解：A 的信息必须“贴合 B 的空间特征”才能落到某个像素位置。
#
# 5) Anti-wrinkle 关键点：末端 refine 阶段只拼 B，不拼 A：
#        concat([out_feat, B]) -> 1x1 conv -> 输出 RGB
#
# 为什么方案3一般比方案1更稳、更不容易带回褶皱？
# - 结构/网格完全由 B 分支确定，A 不具备直接控制空间布局的通路；
# - A 先高通，削弱低频阴影/大尺度明暗（褶皱常见表现）；
# - A 的信息只能通过“受控投影”注入到 B 的网格上，且 refine 不允许把 A 原图 late-fusion 回流。
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

python basicsr/train.py \
  -opt options/train/ljt_train_detail_refine_dualenc.yml \
  --launcher none
