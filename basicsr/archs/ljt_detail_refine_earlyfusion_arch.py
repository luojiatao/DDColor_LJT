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
    """对 3 通道 RGB 做 ImageNet 归一化。"""
    return (x - mean) / std


def _adapt_stem_conv_weight(weight: torch.Tensor, in_chans: int) -> torch.Tensor:
    """把 ConvNeXt stem 的 (out, 3, k, k) 权重扩展到 (out, in_chans, k, k)。

    说明（中文，严谨）：
    - 预训练 ConvNeXt 一般是 3 通道输入。
    - 我们做 early-fusion 会把 B 与 A 在通道维拼接成 6 通道输入。
    - 直接加载预训练会在第一层卷积报 shape mismatch。
    - 常用做法是：把原 3 通道权重复制/平铺到新的通道上，或对额外通道置 0。

    这里采用“平铺复制 + 归一缩放”策略：
    - 若 in_chans 是 3 的倍数（例如 6），则重复 weight 并按重复次数做缩放，保持输出方差大致不变。
    - 否则：前 3 通道复制原权重，其余置 0。

    Args:
        weight: shape = (out_channels, 3, k, k)
        in_chans: 新输入通道数

    Returns:
        new_weight: shape = (out_channels, in_chans, k, k)
    """
    out_c, old_in, k1, k2 = weight.shape
    assert old_in == 3, f"只支持从 3 通道预训练权重适配，收到 old_in={old_in}"  # noqa: E501

    if in_chans == 3:
        return weight

    if in_chans % 3 == 0:
        rep = in_chans // 3
        new_weight = weight.repeat(1, rep, 1, 1) / rep
        return new_weight

    new_weight = torch.zeros((out_c, in_chans, k1, k2), dtype=weight.dtype, device=weight.device)
    new_weight[:, :3, :, :] = weight
    return new_weight


class _EncoderWithHooks(nn.Module):
    """ConvNeXt 编码器 + 多尺度 hook。

    这个类与原 DDColor 的 Encoder 类似，但支持自定义输入通道 in_chans，
    便于 early-fusion 6 通道输入。
    """

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
        # 直接复用原 DDColor 的预训练权重命名
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

        # 关键：适配 stem conv 输入通道数
        stem_key = "downsample_layers.0.0.weight"
        if stem_key in state and state[stem_key].shape[1] != self.in_chans:
            state[stem_key] = _adapt_stem_conv_weight(state[stem_key], self.in_chans)

        incompatible = self.arch.load_state_dict(state, strict=False)
        # 打印缺失/多余 key，方便你排查
        if incompatible.missing_keys:
            print("[Encoder] missing_keys:", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print("[Encoder] unexpected_keys:", incompatible.unexpected_keys)


class _MultiScaleTokenDecoder(nn.Module):
    """与 DDColor 基本一致的多尺度 Transformer 解码器。

    在原论文里是 learnable color tokens；在你的任务里可以理解为 learnable texture tokens。
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

        # 位置编码
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        self.num_heads = nheads
        self.num_layers = dec_layers

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

        # learnable tokens
        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 多尺度 level embedding
        self.num_feature_levels = num_scales
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # input projection，把不同通道数映射到 hidden_dim
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

        # token -> embedding（后面会与像素特征做相关性投影）
        self.token_embed = MLP(hidden_dim, hidden_dim, token_embed_dim, 3)

    def forward(self, feats: list[torch.Tensor], img_features: torch.Tensor) -> torch.Tensor:
        """Args:
        feats: 多尺度特征列表，长度 = num_scales
        img_features: 像素分支输出的空间特征图（决定输出空间网格）
        """
        assert len(feats) == self.num_feature_levels

        src = []
        pos = []
        for i in range(self.num_feature_levels):
            pos_i = self.pe_layer(feats[i], None).flatten(2)
            src_i = self.input_proj[i](feats[i]).flatten(2) + self.level_embed.weight[i][None, :, None]

            # (N,C,HW) -> (HW,N,C)
            pos.append(pos_i.permute(2, 0, 1))
            src.append(src_i.permute(2, 0, 1))

        _, bs, _ = src[0].shape

        # (Q,N,C)
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

        decoder_output = self.decoder_norm(output).transpose(0, 1)  # (Q,N,C)->(N,Q,C)
        token_embed = self.token_embed(decoder_output)

        # 将 token embedding 投影回空间网格（与 DDColor 一致）
        out = torch.einsum("bqc,bchw->bqhw", token_embed, img_features)
        return out


class _Decoder(nn.Module):
    """像素分支(U-Net式上采样) + Token分支(Transformer)，与原 DDColor 的 Decoder 逻辑一致。"""

    def __init__(
        self,
        hooks: list[Hook],
        nf: int = 512,
        blur: bool = True,
        last_norm: str = "Spectral",
        num_queries: int = 100,
        num_scales: int = 3,
        dec_layers: int = 9,
    ):
        super().__init__()
        self.hooks = hooks
        self.nf = nf
        self.blur = blur
        self.last_norm = getattr(NormType, last_norm)

        # 像素分支：用 hooks 的多尺度特征做上采样
        self.layers = self._make_layers()
        embed_dim = nf // 2
        self.last_shuf = CustomPixelShuffle_ICNR(embed_dim, embed_dim, blur=self.blur, norm_type=self.last_norm, scale=4)

        # Token 分支：默认使用 3 个尺度
        # 注意：这里的 in_channels 取决于上采样分支 out0/out1/out2 的通道数，
        # 我们在 init 里用 dummy forward 先跑一遍 encoder+pixel 分支来确定。
        self.num_queries = num_queries
        self.num_scales = num_scales
        self.dec_layers = dec_layers
        if self.num_scales != 3:
            raise ValueError(f"当前实现固定使用3个尺度特征(out0/out1/out2)，请设置 num_scales=3，收到 num_scales={self.num_scales}")
        # 关键：不要在 forward 里懒加载 token decoder。
        # 否则它会在 optimizer 创建之后才出现：
        # - 参数不会加入 optimizer（训练不到）
        # - 默认在 CPU 上创建，导致 CPU/GPU type mismatch
        self._token_decoder: _MultiScaleTokenDecoder | None = None

        # 用 init 时 hooks 里已有的 dummy feature，走一遍像素分支确定 token decoder 的通道维。
        with torch.no_grad():
            encode_feat = self.hooks[-1].feature
            out0 = self.layers[0](encode_feat)
            out1 = self.layers[1](out0)
            out2 = self.layers[2](out1)
            img_features = self.last_shuf(out2)
            self._ensure_token_decoder(out0, out1, out2, img_features)

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

    def _ensure_token_decoder(self, out0: torch.Tensor, out1: torch.Tensor, out2: torch.Tensor, img_features: torch.Tensor) -> None:
        if self._token_decoder is not None:
            return
        in_channels = [out0.shape[1], out1.shape[1], out2.shape[1]]
        self._token_decoder = _MultiScaleTokenDecoder(
            in_channels=in_channels,
            num_queries=self.num_queries,
            num_scales=self.num_scales,
            dec_layers=self.dec_layers,
            token_embed_dim=int(img_features.shape[1]),
        )

    def forward(self) -> torch.Tensor:
        encode_feat = self.hooks[-1].feature
        out0 = self.layers[0](encode_feat)
        out1 = self.layers[1](out0)
        out2 = self.layers[2](out1)
        img_features = self.last_shuf(out2)

        self._ensure_token_decoder(out0, out1, out2, img_features)
        assert self._token_decoder is not None
        out_feat = self._token_decoder([out0, out1, out2], img_features)
        return out_feat


@ARCH_REGISTRY.register()
class DetailRefineEarlyFusion(nn.Module):
    """方案1：单编码器 early-fusion（B 控结构）基线网络。

    设计目标（中文注释，严谨）：
    - 输入：B(无褶皱但模糊) 与 A(有褶皱但清晰) 两张图。
    - 网络：把 B/A 拼成 6 通道输入让 encoder 学融合；decoder 复用 DDColor 的双分支思想。
    - 关键约束：refine 阶段只拼 B，不拼 A，减少 A(褶皱) 的低频信息“捷径回流”。

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

        # 用 dummy forward 让 hooks 里有 feature shape，供 decoder 构造
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

        # refine：只拼 B（3ch），不拼 A
        self.refine_net = nn.Sequential(
            custom_conv_layer(num_queries + 3, out_channels, ks=1, use_activ=False, norm_type=NormType.Spectral)
        )

    def forward(self, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Args:
        b: content/结构图（无褶皱但模糊），shape=(N,3,H,W)
        a: reference/细节图（有褶皱但清晰），shape=(N,3,H,W)
        """
        if self.do_imagenet_normalize:
            b_n = _imagenet_normalize(b, self.mean, self.std)
            a_n = _imagenet_normalize(a, self.mean, self.std)
        else:
            b_n, a_n = b, a

        x = torch.cat([b_n, a_n], dim=1)  # (N,6,H,W)
        self.encoder(x)
        out_feat = self.decoder()  # (N,Q,H',W')

        # 末端只拼 B，减少褶皱回流
        coarse_input = torch.cat([out_feat, b], dim=1)
        out = self.refine_net(coarse_input)
        return out
