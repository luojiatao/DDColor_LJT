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


def _apply_downsample_upsample_blur(img: np.ndarray, scale: float) -> np.ndarray:
    """下采样-上采样模糊：先按比例缩小再放大回原尺寸。
    
    scale: 缩放倍数，例如 2.0 表示先缩小到 1/2 再放大回原尺寸
    """
    if scale <= 1.0:
        return img
    h, w = img.shape[:2]
    new_h, new_w = int(h / scale), int(w / scale)
    # 确保尺寸至少为1
    new_h = max(1, new_h)
    new_w = max(1, new_w)
    # 下采样
    down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # 上采样回原尺寸
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    return up


def _apply_gaussian_blur(img: np.ndarray, radius: int, iterations: int = 1) -> np.ndarray:
    """高斯模糊：指定半径和迭代次数。
    
    radius: 模糊核半径，实际核大小为 2*radius+1
    iterations: 迭代次数，多次应用可增强模糊效果
    """
    if radius <= 0 or iterations <= 0:
        return img
    ksize = 2 * radius + 1
    result = img
    for _ in range(iterations):
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)
    return result


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

        # ========== 模糊数据增强配置（随机选择模式） ==========
        # 总开关：是否启用模糊增强
        self.use_blur_aug = opt.get('use_blur_aug', False)
        # 各增强方式的概率权重 [不做处理, 下采样模糊, 高斯模糊]
        self.blur_aug_probs = opt.get('blur_aug_probs', [0.3, 0.35, 0.35])
        self.blur_aug_target = opt.get('blur_aug_target', 'b')  # 应用目标：'b', 'a', 'both'
        
        # 下采样-上采样模糊参数
        self.downsample_scale = opt.get('downsample_scale', 2.0)  # 缩放倍数，2.0=缩小到1/2再放大
        
        # 高斯模糊参数
        self.gaussian_radius = opt.get('gaussian_radius', 3)  # 核半径，实际核大小=2*radius+1
        self.gaussian_iterations = opt.get('gaussian_iterations', 1)  # 迭代次数

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
        gt = _imread_rgb(gt_path) if gt_path else None
        has_gt = gt is not None

        if self.swap_ab:
            b, a = a, b
            b_path, a_path = a_path, b_path

        # 尺寸校验
        if b.shape[:2] != a.shape[:2]:
            raise ValueError(f"B/A 尺寸不一致: B={b.shape[:2]} A={a.shape[:2]}")
        if has_gt and b.shape[:2] != gt.shape[:2]:
            raise ValueError(f"B/GT 尺寸不一致: B={b.shape[:2]} GT={gt.shape[:2]}")

        # 统一处理图像列表，简化后续分支
        imgs = [b, a, gt] if has_gt else [b, a]

        # 裁剪：gt_size 控制目标尺寸，小于目标则 resize 补齐
        if self.gt_size is not None:
            target_size = int(self.gt_size)
            imgs, actual_crop = _random_crop_same(imgs, target_size)
            if actual_crop < target_size:
                imgs = _resize_same(imgs, target_size)

        # 数据增强：仅 train 阶段
        if self.phase == 'train':
            imgs = augment(imgs, hflip=self.use_hflip, rotation=self.use_rot)
            
            # 随机模糊增强：0=无, 1=下采样模糊, 2=高斯模糊
            if self.use_blur_aug:
                probs = np.array(self.blur_aug_probs, dtype=np.float32)
                probs = probs / probs.sum()
                aug_choice = np.random.choice([0, 1, 2], p=probs)
                
                # 根据 blur_aug_target 确定要处理的索引
                targets = {'b': [0], 'a': [1], 'both': [0, 1]}
                target_indices = targets.get(self.blur_aug_target, [0])
                
                if aug_choice == 1 and self.downsample_scale > 1.0:
                    for i in target_indices:
                        imgs[i] = _apply_downsample_upsample_blur(imgs[i], self.downsample_scale)
                elif aug_choice == 2 and self.gaussian_radius > 0:
                    for i in target_indices:
                        imgs[i] = _apply_gaussian_blur(imgs[i], self.gaussian_radius, self.gaussian_iterations)

        # 解包结果
        b, a = imgs[0], imgs[1]
        gt = imgs[2] if has_gt else a  # 无 GT 时用 A 作占位

        return {
            'b': _to_tensor(b),
            'a': _to_tensor(a),
            'gt': _to_tensor(gt),
            'b_path': b_path,
            'a_path': a_path,
            'gt_path': gt_path if has_gt else a_path,
        }
