import os
from os import path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.ddcolor_arch_utils.unet import (
    Hook,
    CustomPixelShuffle_ICNR,
    UnetBlockWide,
    NormType,
    custom_conv_layer,
)
from basicsr.archs.ddcolor_arch_utils.convnext import ConvNeXt
from basicsr.archs.ddcolor_arch_utils.transformer_utils import (
    SelfAttentionLayer,
    CrossAttentionLayer,
    FFNLayer,
    MLP,
)
from basicsr.archs.ddcolor_arch_utils.position_encoding import PositionEmbeddingSine
from basicsr.utils.registry import ARCH_REGISTRY


def _imagenet_normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def _highpass_avg(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """简单高通：A_hp = A - Blur(A)。

    说明（中文）：
    - 你的约束是“不能带回 A 的褶皱”。褶皱往往包含明显的低频明暗起伏（阴影）。
    - 让参考分支只看到高频（纹理/细节）能显著降低“褶皱低频信息”直接进入网络的概率。
    - 这里用 avg_pool2d 近似模糊，计算简单、可微。
    - 使用 reflection pad 避免边缘黑框。

    Args:
        x: (N,3,H,W)
        k: 模糊核大小，建议奇数（3/5/7）
    """
    if k <= 1:
        return x
    pad = k // 2
    # 使用 reflection padding 避免边缘伪影（原 zero padding 会导致边缘变暗，相减后边缘高亮）
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    blur = F.avg_pool2d(x_pad, kernel_size=k, stride=1, padding=0)
    return x - blur


def _adapt_stem_conv_weight(weight: torch.Tensor, in_chans: int) -> torch.Tensor:
    out_c, old_in, k1, k2 = weight.shape
    assert old_in == 3
    if in_chans == 3:
        return weight
    if in_chans % 3 == 0:
        rep = in_chans // 3
        return weight.repeat(1, rep, 1, 1) / rep
    new_weight = torch.zeros((out_c, in_chans, k1, k2), dtype=weight.dtype, device=weight.device)
    new_weight[:, :3, :, :] = weight
    return new_weight


class _EncoderWithHooks(nn.Module):
    """ConvNeXt 编码器 + hooks，支持任意 in_chans。"""

    def __init__(self, encoder_name: str, hook_names: list[str], in_chans: int = 3, from_pretrain: bool = False):
        super().__init__()

        if encoder_name in {"convnext-t", "convnext"}:
            self.arch = ConvNeXt(in_chans=in_chans)
        elif encoder_name == "convnext-s":
            self.arch = ConvNeXt(in_chans=in_chans, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
        elif encoder_name == "convnext-b":
            self.arch = ConvNeXt(in_chans=in_chans, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        elif encoder_name == "convnext-l":
            self.arch = ConvNeXt(in_chans=in_chans, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        else:
            raise NotImplementedError(f"Unknown encoder_name: {encoder_name}")

        self.encoder_name = encoder_name
        self.in_chans = in_chans
        self.hook_names = hook_names
        self.hooks = [Hook(self.arch._modules[name]) for name in hook_names]

        if from_pretrain:
            self.load_pretrain_model()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.arch(x)

    def load_pretrain_model(self) -> None:
        if self.encoder_name in {"convnext-t", "convnext"}:
            path = "pretrain/convnext_tiny_22k_224.pth"
        elif self.encoder_name == "convnext-s":
            path = "pretrain/convnext_small_22k_224.pth"
        elif self.encoder_name == "convnext-b":
            path = "pretrain/convnext_base_22k_224.pth"
        elif self.encoder_name == "convnext-l":
            path = "pretrain/convnext_large_22k_224.pth"
        else:
            raise NotImplementedError

        # 允许从任意工作目录启动：优先按 DDColor 根目录解析相对路径
        path = osp.expanduser(path)
        if not osp.isabs(path):
            ddcolor_root = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir, osp.pardir, osp.pardir))
            rel = path[2:] if path.startswith("./") else path
            candidate = osp.join(ddcolor_root, rel)
            if osp.isfile(candidate):
                path = candidate
            elif osp.isfile(path):
                path = osp.abspath(path)
            else:
                raise FileNotFoundError(
                    f"找不到预训练权重: {path}\n"
                    f"已尝试: {candidate}\n"
                    f"当前工作目录: {os.getcwd()}"
                )

        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

        stem_key = "downsample_layers.0.0.weight"
        if stem_key in state and state[stem_key].shape[1] != self.in_chans:
            state[stem_key] = _adapt_stem_conv_weight(state[stem_key], self.in_chans)

        incompatible = self.arch.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            print("[Encoder] missing_keys:", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print("[Encoder] unexpected_keys:", incompatible.unexpected_keys)


class _MultiScaleTokenDecoder(nn.Module):
    """多尺度 Transformer token 解码器。

    这里 token 更适合理解为“纹理 token”，memory 来自参考图 A 的高频特征。
    """

    def __init__(
        self,
        in_channels: list[int],
        hidden_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 9,
        pre_norm: bool = False,
        token_embed_dim: int = 256,
        enforce_input_project: bool = True,
        num_scales: int = 3,
    ):
        super().__init__()

        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        self.num_heads = nheads
        self.num_layers = dec_layers
        self.num_feature_levels = num_scales

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm)
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm)
            )
            self.transformer_ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm)
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_levels):
            if in_channels[i] != hidden_dim or enforce_input_project:
                proj = nn.Conv2d(in_channels[i], hidden_dim, kernel_size=1)
                nn.init.kaiming_uniform_(proj.weight, a=1)
                if proj.bias is not None:
                    nn.init.constant_(proj.bias, 0)
                self.input_proj.append(proj)
            else:
                self.input_proj.append(nn.Sequential())

        self.token_embed = MLP(hidden_dim, hidden_dim, token_embed_dim, 3)

    def forward(self, feats_a: list[torch.Tensor], img_features_b: torch.Tensor) -> torch.Tensor:
        assert len(feats_a) == self.num_feature_levels

        src = []
        pos = []
        for i in range(self.num_feature_levels):
            pos_i = self.pe_layer(feats_a[i], None).flatten(2)
            src_i = self.input_proj[i](feats_a[i]).flatten(2) + self.level_embed.weight[i][None, :, None]
            pos.append(pos_i.permute(2, 0, 1))
            src.append(src_i.permute(2, 0, 1))

        _, bs, _ = src[0].shape
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=pos[level_index].to(output.dtype),
                query_pos=query_embed,
            )
            output = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed,
            )
            output = self.transformer_ffn_layers[i](output)

        decoder_output = self.decoder_norm(output).transpose(0, 1)  # (N,Q,C)
        token_embed = self.token_embed(decoder_output)

        # 受控注入核心：A 的 token 只通过与 B 的空间特征相关性投影回去
        out = torch.einsum("bqc,bchw->bqhw", token_embed, img_features_b)
        return out


class _PixelDecoderB(nn.Module):
    """像素分支：只吃 B(结构图) 的 hooks，保证空间网格由 B 决定。"""

    def __init__(self, hooks_b: list[Hook], nf: int = 512, blur: bool = True, last_norm: str = "Spectral"):
        super().__init__()
        self.hooks = hooks_b
        self.nf = nf
        self.blur = blur
        self.last_norm = getattr(NormType, last_norm)

        self.layers = self._make_layers()
        embed_dim = nf // 2
        self.last_shuf = CustomPixelShuffle_ICNR(embed_dim, embed_dim, blur=self.blur, norm_type=self.last_norm, scale=4)

    def _make_layers(self) -> nn.Sequential:
        decoder_layers = []

        e_in_c = self.hooks[-1].feature.shape[1]
        in_c = e_in_c

        out_c = self.nf
        setup_hooks = self.hooks[-2::-1]
        for layer_index, hook in enumerate(setup_hooks):
            feature_c = hook.feature.shape[1]
            if layer_index == len(setup_hooks) - 1:
                out_c = out_c // 2
            decoder_layers.append(
                UnetBlockWide(in_c, feature_c, out_c, hook, blur=self.blur, self_attention=False, norm_type=NormType.Spectral)
            )
            in_c = out_c
        return nn.Sequential(*decoder_layers)

    def forward(self) -> torch.Tensor:
        encode_feat = self.hooks[-1].feature
        out0 = self.layers[0](encode_feat)
        out1 = self.layers[1](out0)
        out2 = self.layers[2](out1)
        img_features_b = self.last_shuf(out2)
        return img_features_b


@ARCH_REGISTRY.register()
class DetailRefineDualEnc(nn.Module):
    """方案3：双编码器（B控结构）+ A 高频纹理记忆 + 受控注入（效果上限优先）。

    严谨点：
    - B 分支决定“空间网格/结构”，因此像素分支只用 B 的特征。
    - A 分支只提供“高频纹理记忆”，并且不允许 A 以原图形式进入 refine（避免褶皱捷径回流）。
    """

    def __init__(
        self,
        encoder_name_b: str = "convnext-l",
        encoder_name_a: str = "convnext-l",
        num_queries: int = 100,
        num_scales: int = 3,
        dec_layers: int = 9,
        nf: int = 512,
        out_channels: int = 3,
        last_norm: str = "Spectral",
        encoder_from_pretrain: bool = True,
        input_size: tuple[int, int] = (256, 256),
        do_imagenet_normalize: bool = True,
        highpass_kernel: int = 5,
    ):
        super().__init__()

        if num_scales != 3:
            raise ValueError(f"当前实现固定使用A分支3个尺度特征(norm3/norm2/norm1)，请设置 num_scales=3，收到 num_scales={num_scales}")

        self.do_imagenet_normalize = do_imagenet_normalize
        self.highpass_kernel = highpass_kernel
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.encoder_b = _EncoderWithHooks(
            encoder_name=encoder_name_b,
            hook_names=["norm0", "norm1", "norm2", "norm3"],
            in_chans=3,
            from_pretrain=encoder_from_pretrain,
        )
        self.encoder_a = _EncoderWithHooks(
            encoder_name=encoder_name_a,
            hook_names=["norm0", "norm1", "norm2", "norm3"],
            in_chans=3,
            from_pretrain=encoder_from_pretrain,
        )

        # dummy forward，让 hooks 里有 feature
        self.encoder_b.eval()
        self.encoder_a.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, input_size[0], input_size[1])
            self.encoder_b(dummy)
            self.encoder_a(dummy)
        self.encoder_b.train()
        self.encoder_a.train()

        # B 的像素分支（决定空间网格）
        self.pixel_decoder_b = _PixelDecoderB(self.encoder_b.hooks, nf=nf, last_norm=last_norm)

        # A 的 token 分支（memory 来自 A 的多尺度特征）
        # 取 A 的 3 个尺度特征作为 memory（从低分辨率到相对高分辨率）
        # 这里选择 norm3/norm2/norm1 三层（最深到较浅），更偏语义+纹理。
        feats_a = [
            self.encoder_a.hooks[-1].feature,
            self.encoder_a.hooks[-2].feature,
            self.encoder_a.hooks[-3].feature,
        ]
        in_channels_a = [t.shape[1] for t in feats_a]
        self.token_decoder = _MultiScaleTokenDecoder(
            in_channels=in_channels_a,
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
            token_embed_dim=int(nf // 2),
        )

        # refine：只拼 B（结构锚点）
        self.refine_net = nn.Sequential(
            custom_conv_layer(num_queries + 3, out_channels, ks=1, use_activ=False, norm_type=NormType.Spectral)
        )

    def forward(self, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # 1) 归一化
        if self.do_imagenet_normalize:
            b_n = _imagenet_normalize(b, self.mean, self.std)
            a_n = _imagenet_normalize(a, self.mean, self.std)
        else:
            b_n, a_n = b, a

        # 2) A 高频化，尽量只保留纹理（削弱褶皱低频阴影）
        a_hp = _highpass_avg(a_n, k=self.highpass_kernel)

        # 3) 编码
        self.encoder_b(b_n)
        self.encoder_a(a_hp)

        # 4) B 像素分支输出空间特征（空间网格由 B 决定）
        img_features_b = self.pixel_decoder_b()

        # 5) A 作为 memory 的多尺度特征（受控注入）
        feats_a = [
            self.encoder_a.hooks[-1].feature,
            self.encoder_a.hooks[-2].feature,
            self.encoder_a.hooks[-3].feature,
        ]
        out_feat = self.token_decoder(feats_a, img_features_b)

        # 6) refine：只拼 B，不拼 A
        coarse_input = torch.cat([out_feat, b], dim=1)
        out = self.refine_net(coarse_input)
        return out
