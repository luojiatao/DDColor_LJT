"""ResUnet 细节修复网络架构（轻量级方案）。

用于 A/B→C 细节修复任务：
- B(结构/内容图)：无褶皱但模糊
- A(参考/细节图)：有褶皱但清晰  
- C(目标)：无褶皱且清晰

相比 DDColorMask 方案，本方案使用 ResNet34 + U-Net 结构，更轻量。
"""

import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY


def _imagenet_normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """对 3 通道 RGB 做 ImageNet 归一化。"""
    return (x - mean) / std


def _imagenet_denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """ImageNet 反归一化。"""
    return x * std + mean


def _get_resunet6c(n_out: int, pretrained: bool):
    """延迟导入 ResUnet6C，避免在模块加载时触发 mmcv 依赖。
    
    与 ResUnet6CDetailTransfer 保持一致：
    - img_size 固定为 (512, 512)
    - 使用 init_net_exclude 初始化非 backbone 层
    """
    import sys
    from os import path as osp
    
    # 添加项目根目录到 path
    _root = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir, osp.pardir))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    
    from cjy_res_unet.networks import ResUnet6C, resnet34, init_net_exclude
    from fastai.vision.all import NormType
    
    # 与 res_unet_model.py 保持一致：img_size=(512,512)
    img_size = torch.Size([512, 512])
    
    net = ResUnet6C(
        arch=resnet34,
        n_out=n_out,
        img_size=img_size,
        pretrained=pretrained,
        blur=True,
        norm_type=NormType.Weight,
        self_attention=False,
    )
    
    # 与 res_unet_model.py 保持一致：对非 backbone 层进行 normal 初始化
    net = init_net_exclude(net, 'normal', init_gain=0.02, gpu_ids=[], exclude_layers=[0])
    
    return net


@ARCH_REGISTRY.register()
class DetailRefineResUnet(nn.Module):
    """ResUnet 细节修复网络：Early-Fusion 6 通道输入。

    网络结构：
    - 输入：B(结构/内容图) 与 A(参考/细节图)
    - Early-Fusion：concat(B, A) => 6ch 输入 ResNet34 编码器
    - U-Net 解码器：跳跃连接 + 上采样
    - 输出模式：直接 RGB / 残差 / 残差+Mask

    forward(b, a) -> rgb 输出（默认 3 通道）。
    """

    def __init__(
        self,
        out_channels: int = 3,
        output_mode: str = "residual_and_mask",
        input_size=(256, 256),
        pretrained_backbone: bool = True,
    ):
        """
        Args:
            out_channels: 输出通道数，默认 3（RGB）
            output_mode: 输出模式
                - "rgb": 直接输出 RGB
                - "residual": 输出残差，最终 = 残差 + content
                - "residual_and_mask": 输出 4 通道（3ch 残差 + 1ch mask），最终 = 残差 * mask + content
            input_size: 输入图像尺寸（用于初始化）
            pretrained_backbone: 是否使用 ImageNet 预训练的 ResNet34
        """
        super().__init__()

        self.output_mode = output_mode
        self.out_channels = out_channels

        # 根据输出模式确定 U-Net 输出通道数
        if output_mode == "residual_and_mask":
            unet_out_channels = 4  # 3ch RGB residual + 1ch mask
        else:
            unet_out_channels = out_channels

        # 构建 ResUnet6C（6 通道输入）- 延迟导入避免 mmcv 依赖
        # input_size 参数已废弃，内部固定使用 (512, 512)
        self.unet = _get_resunet6c(unet_out_channels, pretrained_backbone)

        # ImageNet 归一化参数
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self, b: torch.Tensor, a: torch.Tensor, return_intermediate: bool = False
    ):
        """
        Args:
            b: content/结构图，shape=(N,3,H,W)，范围 [0,1]
               对应 ResUnet6CDetailTransfer 的 content_images（无褶皱但模糊）
            a: style/参考图，shape=(N,3,H,W)，范围 [0,1]
               对应 ResUnet6CDetailTransfer 的 style_images（有褶皱但清晰）
            return_intermediate: 是否返回中间结果（残差/mask），用于可视化调试
        Returns:
            - 默认：输出 RGB 图像，shape=(N,3,H,W)，范围 [0,1]
            - return_intermediate=True 时返回 dict:
                - "output": 最终 RGB
                - "residual": 残差图（仅 residual/residual_and_mask 模式）
                - "mask": mask 图（仅 residual_and_mask 模式）
        
        与 ResUnet6CDetailTransfer 的对应关系：
            b ↔ content_images（结构/内容图）
            a ↔ style_images（参考/细节图）
        """
        # ImageNet 归一化
        b_n = _imagenet_normalize(b, self.mean, self.std)
        a_n = _imagenet_normalize(a, self.mean, self.std)

        # Early-Fusion：拼接成 6 通道
        x = torch.cat([b_n, a_n], dim=1)  # (N,6,H,W)

        # U-Net 前向
        raw_out = self.unet(x)

        # 中间结果容器
        intermediate: dict[str, torch.Tensor] = {}

        # 根据输出模式处理
        if self.output_mode == "residual":
            # 残差模式：输出 = 残差 + content（归一化空间）
            residual = raw_out  # 3ch 残差（归一化空间）
            out = residual + b_n
            out = _imagenet_denormalize(out, self.mean, self.std)
            intermediate["residual"] = _imagenet_denormalize(residual, self.mean, self.std)
        elif self.output_mode == "residual_and_mask":
            # 残差+Mask 模式：输出 = 残差 * sigmoid(mask) + content
            diff_rgb = raw_out[:, :3, :, :]
            mask = torch.sigmoid(raw_out[:, 3:4, :, :])
            out = diff_rgb * mask + b_n
            out = _imagenet_denormalize(out, self.mean, self.std)
            intermediate["residual"] = _imagenet_denormalize(diff_rgb, self.mean, self.std)
            intermediate["mask"] = mask
        else:
            # 直接 RGB 模式
            out = _imagenet_denormalize(raw_out, self.mean, self.std)

        if return_intermediate:
            intermediate["output"] = out
            return intermediate
        return out
