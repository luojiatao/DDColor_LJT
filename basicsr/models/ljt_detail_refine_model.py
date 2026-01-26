import os
from collections import OrderedDict
from os import path as osp

import torch

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import master_only
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class DetailRefineModel(BaseModel):
    """用于 A/B→C 细节修复的训练模型封装。

    输入（dataset 返回）：
    - b: 结构图（无褶皱但模糊）
    - a: 参考图（有褶皱但清晰）
    - gt: 目标真值（无褶皱且清晰）

    说明（中文，严谨）：
    - 这里按“监督学习”实现：直接拟合输出到 gt。
    - 默认只用像素/感知损失；若你在 yaml 里配置 gan_opt 和 network_d，也支持对抗训练。
    """

    def __init__(self, opt):
        super().__init__(opt)

        # net_g
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained generator
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # 可选：net_d（GAN）
        self.use_gan = False
        self.net_d = None
        if self.is_train and self.opt.get('network_d') is not None and self.opt['train'].get('gan_opt') is not None:
            from basicsr.archs import build_network as build_arch

            self.net_d = build_arch(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)

            load_path_d = self.opt['path'].get('pretrain_network_d', None)
            if load_path_d is not None:
                param_key = self.opt['path'].get('param_key_d', 'params')
                self.load_network(self.net_d, load_path_d, self.opt['path'].get('strict_load_d', True), param_key)

            self.use_gan = True

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.net_g.train()
        if self.use_gan:
            self.net_d.train()

        # losses
        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device) if train_opt.get('pixel_opt') else None
        self.cri_perceptual = (
            build_loss(train_opt['perceptual_opt']).to(self.device) if train_opt.get('perceptual_opt') else None
        )
        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device) if (self.use_gan and train_opt.get('gan_opt')) else None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_gan is None:
            raise ValueError('至少需要配置 pixel_opt / perceptual_opt / gan_opt 其中之一。')

        # optimizers
        self.optimizers = []
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        if self.use_gan:
            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)

        # schedulers
        self.setup_schedulers()

    def feed_data(self, data):
        self.b = data['b'].to(self.device)
        self.a = data['a'].to(self.device)
        self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        # ===== G =====
        if self.use_gan:
            for p in self.net_d.parameters():
                p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.b, self.a)

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix is not None:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_g_pix'] = l_pix

        if self.cri_perceptual is not None:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_g_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_g_style'] = l_style

        if self.use_gan and self.cri_gan is not None:
            fake_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_pred, target_is_real=True, is_disc=False)
            l_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        l_total.backward()
        self.optimizer_g.step()

        # ===== D =====
        if self.use_gan:
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            real_pred = self.net_d(self.gt)
            fake_pred = self.net_d(self.output.detach())
            l_d = self.cri_gan(real_pred, target_is_real=True, is_disc=True) + self.cri_gan(
                fake_pred, target_is_real=False, is_disc=True
            )
            l_d.backward()
            self.optimizer_d.step()

            loss_dict['l_d'] = l_d
            loss_dict['real_score'] = real_pred.detach().mean()
            loss_dict['fake_score'] = fake_pred.detach().mean()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.b, self.a)
        self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['b'] = self.b.detach().cpu()
        out_dict['a'] = self.a.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # 与项目内其他模型一致：分布式验证只在 rank0 执行
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        dataset_phase = dataloader.dataset.opt.get('phase', 'val')
        
        # 测试集（phase='test'）或无 metrics 配置时，跳过 metrics 计算
        is_test_set = dataset_phase == 'test'
        with_metrics = (not is_test_set) and self.opt.get('val') is not None and self.opt['val'].get('metrics') is not None
        use_pbar = self.opt.get('val', {}).get('pbar', False)

        if with_metrics and not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        if with_metrics:
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        pbar = None
        if use_pbar:
            try:
                from tqdm import tqdm  # type: ignore

                pbar = tqdm(total=len(dataloader), unit='image')
            except Exception:
                pbar = None

        for idx, val_data in enumerate(dataloader):
            # 取文件名作为样本名：优先用 b_path（因为 test 集可能无真实 GT）
            if 'b_path' in val_data:
                img_name = osp.splitext(osp.basename(val_data['b_path'][0]))[0]
            elif 'gt_path' in val_data:
                img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            else:
                img_name = f'{idx:08d}'

            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            gt_img = tensor2img([visuals['gt']])
            metric_data['img2'] = gt_img

            # 释放显存碎片（不影响正确性）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_dir = osp.join(self.opt['path']['visualization'], img_name)
                    for key in visuals:
                        save_path = osp.join(save_dir, f'{current_iter}_{key}.png')
                        img = tensor2img(visuals[key])
                        imwrite(img, save_path)
                else:
                    suffix = self.opt.get('val', {}).get('suffix', '')
                    if suffix:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{suffix}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{self.opt["name"]}.png')
                    imwrite(sr_img, save_img_path)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f'Val {img_name}')

        if pbar is not None:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            log_str = f'Validation {dataset_name}\n'
            for metric, value in self.metric_results.items():
                log_str += f'\t # {metric}: {value:.4f}'
                if hasattr(self, 'best_metric_results'):
                    log_str += (
                        f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                        f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                    )
                log_str += '\n'

            logger = get_root_logger()
            logger.info(log_str)
            if tb_logger:
                for metric, value in self.metric_results.items():
                    tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def _export_comfy_state_dict(self, current_iter: int, net_label: str = 'net_g', net=None):
        """导出 ComfyUI 可直接加载的权重文件（纯 state_dict，无外层包裹）。

        ComfyUI 节点加载方式：model.load_state_dict(torch.load(model_path), strict=True)
        因此需要：
        - 纯 state_dict（不要外层 {'params': ...} 包裹）
        - 去掉 DDP 的 'module.' 前缀
        - 兼容命名差异：训练端 decoder 可能是 `_token_decoder`/`token_embed`，
          而推理端实现常用 `color_decoder`/`color_embed`
        """
        if net is None:
            net = self.net_g

        iter_tag = 'latest' if current_iter == -1 else str(current_iter)
        save_filename = f'{net_label}_{iter_tag}.pth'
        save_path = osp.join(self.opt['path']['models'], save_filename)

        bare_net = self.get_bare_model(net)
        sd = bare_net.state_dict()

        out = OrderedDict()
        for k, v in sd.items():
            key = k
            if key.startswith('module.'):
                key = key[7:]

            # 训练端：decoder._token_decoder.* / *.token_embed.*
            # 推理端：decoder.color_decoder.* / *.color_embed.*
            if key.startswith('decoder._token_decoder.'):
                key = key.replace('decoder._token_decoder.', 'decoder.color_decoder.', 1)
            key = key.replace('.token_embed.', '.color_embed.')

            out[key] = v.detach().cpu()

        # 重试写入（避免偶发 IO 错误）
        retry = 3
        while retry > 0:
            try:
                torch.save(out, save_path)
                break
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'[export_comfy] Save model error: {e}, remaining retry times: {retry - 1}')
                import time as _time
                _time.sleep(1)
                retry -= 1
        if retry == 0:
            logger = get_root_logger()
            logger.warning(f'[export_comfy] Still cannot save {save_path}. Just ignore it.')

    @master_only
    def save(self, epoch, current_iter):
        # 只保存 ComfyUI 格式权重（纯 state_dict，无外层包裹）
        self._export_comfy_state_dict(current_iter, 'net_g')
        if self.use_gan:
            self._export_comfy_state_dict(current_iter, 'net_d', net=self.net_d)
        self.save_training_state(epoch, current_iter)
