import torch
import torch.nn as nn

from basicsr.archs.ddcolor_arch_utils.unet import NormType, custom_conv_layer
from basicsr.utils.registry import ARCH_REGISTRY

# 复用 EarlyFusion 方案里已经实现好的：
# - 6 通道 early-fusion ConvNeXt encoder（含预训练权重适配）
# - DDColor 风格的 decoder（像素分支 + token/queries 分支）
from .ljt_detail_refine_earlyfusion_arch import _Decoder, _EncoderWithHooks, _imagenet_normalize


@ARCH_REGISTRY.register()
class DetailRefineDDColorMask(nn.Module):
    """DDColorMask 复现版：Early-Fusion + refine 同时拼 B 与 A。

    对应你在 cjy_ddcolor 目录下的 ddcolormask_arch_cjy_detail_transfer.py 的核心结构：
    - 输入：B(结构/内容图) 与 A(参考/细节图)
    - Early-Fusion：concat(B, A) => 6ch 输入单个 ConvNeXt 编码器
    - Decoder：复用 DDColor 的“像素上采样 + token queries”结构，输出 Q 个特征图 out_feat
    - Refine：concat(out_feat, B, A) => 1x1 conv 输出 RGB

    与方案1(EarlyFusion 基线) 的关键差异：
    - EarlyFusion 基线的 refine 只拼 B（避免 A 褶皱捷径回流）
    - 本方案的 refine 同时拼 B 与 A（更强 late-fusion，更接近 DDColorMask 原实现）

    forward(b, a) -> rgb 输出（默认 3 通道）。
    """

    def __init__(
        self,
        encoder_name: str = "convnext-l",
        num_queries: int = 100,
        num_scales: int = 3,
        dec_layers: int = 9,
        nf: int = 512,
        out_channels: int = 3,
        last_norm: str = "Spectral",
        encoder_from_pretrain: bool = True,
        input_size: tuple[int, int] = (256, 256),
        do_imagenet_normalize: bool = True,
    ):
        super().__init__()

        self.do_imagenet_normalize = do_imagenet_normalize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 单编码器：输入为 concat(B,A) => 6 通道
        self.encoder = _EncoderWithHooks(
            encoder_name=encoder_name,
            hook_names=["norm0", "norm1", "norm2", "norm3"],
            in_chans=6,
            from_pretrain=encoder_from_pretrain,
        )

        # dummy forward：让 hooks feature shape 就绪
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 6, input_size[0], input_size[1])
            self.encoder(dummy)
        self.encoder.train()

        self.decoder = _Decoder(
            hooks=self.encoder.hooks,
            nf=nf,
            last_norm=last_norm,
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
        )

        # refine：拼 B 与 A（DDColorMask 风格）
        self.refine_net = nn.Sequential(
            custom_conv_layer(num_queries + 6, out_channels, ks=1, use_activ=False, norm_type=NormType.Spectral)
        )

    def forward(self, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Args:
        b: content/结构图，shape=(N,3,H,W)
        a: style/参考图，shape=(N,3,H,W)
        """

        if self.do_imagenet_normalize:
            b_n = _imagenet_normalize(b, self.mean, self.std)
            a_n = _imagenet_normalize(a, self.mean, self.std)
        else:
            b_n, a_n = b, a

        x = torch.cat([b_n, a_n], dim=1)  # (N,6,H,W)
        self.encoder(x)
        out_feat = self.decoder()  # (N,Q,H',W')

        coarse_input = torch.cat([out_feat, b, a], dim=1)
        out = self.refine_net(coarse_input)
        return out
