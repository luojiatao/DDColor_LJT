"""ResUnet 细节修复训练模型（支持中间结果可视化）。

继承自 DetailRefineModel，扩展以下功能：
- 验证/测试时可视化：输出图、残差图、Mask 图
- 支持 residual_and_mask 模式的完整调试信息

使用方法：在 yml 配置中设置 model_type: DetailRefineResUnetModel
"""

from collections import OrderedDict
from os import path as osp

import torch

from basicsr.utils import imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .ljt_detail_refine_model import DetailRefineModel


@MODEL_REGISTRY.register()
class DetailRefineResUnetModel(DetailRefineModel):
    """ResUnet 细节修复模型，支持中间结果（残差/Mask）可视化。

    相比 DetailRefineModel，主要变化：
    1. test() 时调用 return_intermediate=True 获取残差和 mask
    2. get_current_visuals() 返回完整的可视化信息
    3. 验证保存时输出更丰富的调试图像
    """

    def __init__(self, opt):
        super().__init__(opt)
        # 缓存中间结果
        self._intermediate: dict[str, torch.Tensor] = {}

    def test(self):
        """推理时获取中间结果（残差/Mask）。"""
        self.net_g.eval()
        with torch.no_grad():
            result = self.net_g(self.b, self.a, return_intermediate=True)
            
            # 兼容两种返回格式
            if isinstance(result, dict):
                self.output = result['output']
                self._intermediate = result
            else:
                # 旧版网络或 rgb 模式
                self.output = result
                self._intermediate = {'output': result}
        self.net_g.train()

    def get_current_visuals(self):
        """返回完整的可视化结果。

        Returns:
            OrderedDict 包含：
            - b: 输入结构图（无褶皱但模糊）
            - a: 输入参考图（有褶皱但清晰）
            - result: 网络输出
            - gt: 真值
            - residual: 残差图（仅 residual/residual_and_mask 模式）
            - mask: Mask 图（仅 residual_and_mask 模式）
        """
        out_dict = OrderedDict()
        out_dict['b'] = self.b.detach().cpu()
        out_dict['a'] = self.a.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['gt'] = self.gt.detach().cpu()

        # 添加中间结果
        if 'residual' in self._intermediate:
            out_dict['residual'] = self._intermediate['residual'].detach().cpu()
        if 'mask' in self._intermediate:
            # mask 是单通道，扩展为 3 通道便于可视化
            mask = self._intermediate['mask'].detach().cpu()
            out_dict['mask'] = mask.expand(-1, 3, -1, -1)

        return out_dict

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """验证逻辑，支持保存中间结果。"""
        from basicsr.metrics import calculate_metric
        from basicsr.utils import get_root_logger

        dataset_name = dataloader.dataset.opt['name']
        dataset_phase = dataloader.dataset.opt.get('phase', 'val')

        is_test_set = dataset_phase == 'test'
        with_metrics = (not is_test_set) and self.opt.get('val') is not None and self.opt['val'].get('metrics') is not None
        use_pbar = self.opt.get('val', {}).get('pbar', False)

        if with_metrics and not hasattr(self, 'metric_results'):
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        if with_metrics:
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        pbar = None
        if use_pbar:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(dataloader), unit='image')
            except Exception:
                pbar = None

        for idx, val_data in enumerate(dataloader):
            # 取文件名
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

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    # 训练期间验证：保存到 visualization 目录
                    save_dir = osp.join(self.opt['path']['visualization'], img_name)
                    self._save_visuals_with_intermediate(visuals, save_dir, current_iter)
                else:
                    # 纯测试：保存完整结果
                    save_dir = osp.join(self.opt['path']['visualization'], dataset_name, img_name)
                    self._save_visuals_with_intermediate(visuals, save_dir, current_iter)

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

                # 额外：将中间结果写入 TensorBoard
                self._log_intermediate_to_tb(tb_logger, visuals, current_iter, dataset_name)

    def _save_visuals_with_intermediate(self, visuals: OrderedDict, save_dir: str, current_iter: int):
        """保存完整可视化结果，包括残差和 Mask。"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        iter_tag = f'{current_iter}_' if current_iter > 0 else ''

        # 保存所有可视化项
        for key, tensor in visuals.items():
            save_path = osp.join(save_dir, f'{iter_tag}{key}.png')
            img = tensor2img(tensor)
            imwrite(img, save_path)

        # 额外：生成对比拼图（b | result | gt）
        try:
            import numpy as np
            b_img = tensor2img(visuals['b'])
            result_img = tensor2img(visuals['result'])
            gt_img = tensor2img(visuals['gt'])
            compare = np.concatenate([b_img, result_img, gt_img], axis=1)
            compare_path = osp.join(save_dir, f'{iter_tag}compare_b_result_gt.png')
            imwrite(compare, compare_path)
        except Exception:
            pass  # 拼图失败不影响主流程

    def _log_intermediate_to_tb(self, tb_logger, visuals: OrderedDict, current_iter: int, dataset_name: str):
        """将中间结果写入 TensorBoard。"""
        try:
            # 只取 batch 中第一张
            for key in ['result', 'residual', 'mask']:
                if key in visuals:
                    tensor = visuals[key]
                    if tensor.dim() == 4:
                        tensor = tensor[0]  # (C,H,W)
                    # 确保值在 [0,1]
                    tensor = tensor.clamp(0, 1)
                    tb_logger.add_image(f'{dataset_name}/{key}', tensor, current_iter)
        except Exception:
            pass  # TensorBoard 日志失败不影响训练
