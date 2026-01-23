import os
from os import path as osp

import cv2
import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.transforms import augment
from basicsr.utils.registry import DATASET_REGISTRY


def _imread_rgb(path: str) -> np.ndarray:
    """读取 RGB 图像，返回 float32, range [0, 1]。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC RGB [0,1] -> CHW tensor [0,1]"""
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


def _random_crop_same(imgs: list[np.ndarray], crop_size: int) -> list[np.ndarray]:
    """对多张同尺寸图做一致随机裁剪。"""
    h, w = imgs[0].shape[:2]
    if h < crop_size or w < crop_size:
        raise ValueError(f"图像尺寸({h},{w})小于crop_size={crop_size}")
    top = np.random.randint(0, h - crop_size + 1)
    left = np.random.randint(0, w - crop_size + 1)
    out = [im[top : top + crop_size, left : left + crop_size, :] for im in imgs]
    return out


@DATASET_REGISTRY.register()
class TripletRefineDataset(data.Dataset):
    """三元组数据集：B(结构/无褶皱模糊), A(参考/有褶皱清晰), C(GT/无褶皱清晰)。

    严谨说明：
    - 训练必须对三张图做“完全一致”的数据增强（crop/flip/rotate），否则监督信号会失效。
    - 建议用 meta_info_file 显式列出三张图路径，避免靠文件名规则隐式匹配。

    meta_info_file 格式（每行一条样本）：
        path_to_B path_to_A path_to_C
    支持空格或制表符分隔。

    返回 dict:
        {
          'b': Tensor(3,H,W),
          'a': Tensor(3,H,W),
          'gt': Tensor(3,H,W),
          'b_path': str,
          'a_path': str,
          'gt_path': str,
        }
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.phase = opt['phase']

        # 是否交换 A/B：用于对齐某些推理侧实现的输入语义。
        # 默认 False：保持 meta_info_file 的列顺序 (B, A, GT)。
        self.swap_ab = bool(opt.get('swap_ab', False))

        self.gt_size = opt.get('gt_size', None)
        self.use_hflip = opt.get('use_hflip', True)
        self.use_rot = opt.get('use_rot', False)

        meta = opt.get('meta_info_file')
        if meta is None:
            raise ValueError("TripletRefineDataset 需要在 yaml 中提供 meta_info_file")

        if isinstance(meta, (list, tuple)):
            meta_files = meta
        else:
            meta_files = [meta]

        self.samples: list[tuple[str, str, str]] = []
        for mf in meta_files:
            mf = osp.expanduser(mf)
            if not osp.isfile(mf):
                raise FileNotFoundError(f"meta_info_file 不存在: {mf}")
            with open(mf, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        raise ValueError(f"meta 行格式错误（需要3列路径）: {line}")
                    b_path, a_path, gt_path = parts[0], parts[1], parts[2]
                    self.samples.append((b_path, a_path, gt_path))

        if len(self.samples) == 0:
            raise ValueError("meta_info_file 为空，未读取到样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b_path, a_path, gt_path = self.samples[idx]

        b = _imread_rgb(b_path)
        a = _imread_rgb(a_path)
        gt = _imread_rgb(gt_path)

        if self.swap_ab:
            b, a = a, b
            b_path, a_path = a_path, b_path

        # 基本校验：必须同尺寸（否则需要你先做配准/resize）
        if b.shape[:2] != a.shape[:2] or b.shape[:2] != gt.shape[:2]:
            raise ValueError(
                f"三元组尺寸不一致: B={b.shape[:2]} A={a.shape[:2]} GT={gt.shape[:2]}\n"
                f"B={b_path}\nA={a_path}\nGT={gt_path}"
            )

        if self.phase == 'train':
            if self.gt_size is not None:
                b, a, gt = _random_crop_same([b, a, gt], int(self.gt_size))
            b, a, gt = augment([b, a, gt], hflip=self.use_hflip, rotation=self.use_rot)
        else:
            # val/test：默认不做随机增强；如需 center crop 可自行加配置
            pass

        return {
            'b': _to_tensor(b),
            'a': _to_tensor(a),
            'gt': _to_tensor(gt),
            'b_path': b_path,
            'a_path': a_path,
            'gt_path': gt_path,
        }
