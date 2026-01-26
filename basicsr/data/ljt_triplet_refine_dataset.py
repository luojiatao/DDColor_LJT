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


def _random_crop_same(imgs: list[np.ndarray], crop_size: int) -> tuple[list[np.ndarray], int]:
    """对多张同尺寸图做一致随机裁剪。
    
    如果图像的长或宽小于 crop_size，则按图像的较小边裁剪。
    例如：crop_size=512，原图 418x768 → 裁剪为 418x418
    
    返回：(裁剪后的图像列表, 实际裁剪尺寸)
    """
    h, w = imgs[0].shape[:2]
    # 实际裁剪尺寸 = min(目标尺寸, 图像较小边)
    actual_crop = min(crop_size, h, w)
    top = np.random.randint(0, h - actual_crop + 1)
    left = np.random.randint(0, w - actual_crop + 1)
    out = [im[top : top + actual_crop, left : left + actual_crop, :] for im in imgs]
    return out, actual_crop


def _resize_same(imgs: list[np.ndarray], target_size: int) -> list[np.ndarray]:
    """将多张图 resize 到统一尺寸（正方形）。"""
    out = []
    for im in imgs:
        resized = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        out.append(resized)
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
                    if len(parts) == 2:
                        # 两列格式（test 无 GT）：B, A
                        b_path, a_path = parts[0], parts[1]
                        gt_path = None  # 无 GT
                    elif len(parts) >= 3:
                        # 三列格式：B, A, GT
                        b_path, a_path, gt_path = parts[0], parts[1], parts[2]
                    else:
                        raise ValueError(f"meta 行格式错误（需要2或3列路径）: {line}")
                    self.samples.append((b_path, a_path, gt_path))

        if len(self.samples) == 0:
            raise ValueError("meta_info_file 为空，未读取到样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b_path, a_path, gt_path = self.samples[idx]

        b = _imread_rgb(b_path)
        a = _imread_rgb(a_path)
        
        # GT 可能为 None（test 阶段无 GT）
        if gt_path is not None:
            gt = _imread_rgb(gt_path)
        else:
            gt = None

        if self.swap_ab:
            b, a = a, b
            b_path, a_path = a_path, b_path

        # 基本校验：B 和 A 必须同尺寸
        if b.shape[:2] != a.shape[:2]:
            raise ValueError(
                f"B/A 尺寸不一致: B={b.shape[:2]} A={a.shape[:2]}\n"
                f"B={b_path}\nA={a_path}"
            )
        if gt is not None and b.shape[:2] != gt.shape[:2]:
            raise ValueError(
                f"三元组尺寸不一致: B={b.shape[:2]} A={a.shape[:2]} GT={gt.shape[:2]}\n"
                f"B={b_path}\nA={a_path}\nGT={gt_path}"
            )

        # 裁剪：train/val/test 都支持，由 gt_size 控制
        # 裁剪后如果尺寸小于 gt_size，会 resize 到 gt_size 保证 batch 内尺寸一致
        if self.gt_size is not None:
            target_size = int(self.gt_size)
            if gt is not None:
                [b, a, gt], actual_crop = _random_crop_same([b, a, gt], target_size)
                if actual_crop < target_size:
                    b, a, gt = _resize_same([b, a, gt], target_size)
            else:
                [b, a], actual_crop = _random_crop_same([b, a], target_size)
                if actual_crop < target_size:
                    b, a = _resize_same([b, a], target_size)

        # 数据增强：仅 train 阶段
        if self.phase == 'train':
            if gt is not None:
                b, a, gt = augment([b, a, gt], hflip=self.use_hflip, rotation=self.use_rot)
            else:
                b, a = augment([b, a], hflip=self.use_hflip, rotation=self.use_rot)

        result = {
            'b': _to_tensor(b),
            'a': _to_tensor(a),
            'b_path': b_path,
            'a_path': a_path,
        }
        
        if gt is not None:
            result['gt'] = _to_tensor(gt)
            result['gt_path'] = gt_path
        else:
            # 无 GT 时用 A 作为占位（仅用于 feed_data 不报错，不参与 loss 计算）
            result['gt'] = _to_tensor(a)
            result['gt_path'] = a_path
        
        return result
