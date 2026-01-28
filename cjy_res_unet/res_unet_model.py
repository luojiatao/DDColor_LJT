from .networks import *
import yaml


class ResUnet6CDetailTransfer(nn.Module):
    def __init__(self, cfg_yaml_path):
        super(ResUnet6CDetailTransfer, self).__init__()
        with open(cfg_yaml_path, 'r') as file:
            config = yaml.safe_load(file)
            print('init ddcolor with cfg', cfg_yaml_path, config)
            self.output_mode = config['output_mode'] if 'output_mode' in config else 'rgb'
            self.out_chanel = config['out_chanel'] if 'out_chanel' in config else 3

        arch = resnet34
        img_size = torch.Size([512, 512])
        net = ResUnet6C(arch, self.out_chanel, img_size, pretrained=True, blur=True,
                        norm_type=NormType.Weight, self_attention=False)
        # 不初始化backbone部分
        self.unet = init_net_exclude(net, 'normal', init_gain=0.02,
                                     gpu_ids=[], exclude_layers=[0])

        device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def forward(self, content_images, style_images):
        content_images_imagenet_norm = self.normalize(content_images)
        style_images_imagenet_norm = self.normalize(style_images)
        x = torch.cat([content_images_imagenet_norm, style_images_imagenet_norm], dim=1)
        out = self.unet(x)
        diff_mask = None
        out_diff_rgb = None
        if self.output_mode == 'residual':
            out = out + content_images_imagenet_norm
            out = self.denormalize(out)
        elif self.output_mode == 'residual_and_mask':
            assert out.shape[1] == 4, "For 'residual_and_mask' output_mode, the output channel must be 4 (3 for RGB residual + 1 for mask)."
            diff_mask = out[:, -1:, :, :]
            diff_mask = torch.sigmoid(diff_mask)
            out_diff_rgb = out[:, :3, :, :]
            out = out_diff_rgb * diff_mask + content_images_imagenet_norm
            out = self.denormalize(out)
        else:
            out = self.denormalize(out)
        return {
            "output": out,
            "diff_mask": diff_mask,
            "out_diff_rgb": out_diff_rgb,
        }

    def postprocess(self, output):
        return output

    def init_weights(self, net, init_type='normal', init_gain=0.02):
        pass
        # """Initialize network weights.
        #
        # Parameters:
        #     net (network)   -- network to be initialized
        #     init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        #     init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
        #
        # We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        # work better for some applications. Feel free to try yourself.
        # """
        #
        # def init_func(m):  # define the initialization function
        #     classname = m.__class__.__name__
        #     if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        #         if init_type == 'normal':
        #             init.normal_(m.weight.data, 0.0, init_gain)
        #         elif init_type == 'xavier':
        #             init.xavier_normal_(m.weight.data, gain=init_gain)
        #         elif init_type == 'kaiming':
        #             init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        #         elif init_type == 'orthogonal':
        #             init.orthogonal_(m.weight.data, gain=init_gain)
        #         else:
        #             raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        #         if hasattr(m, 'bias') and m.bias is not None:
        #             init.constant_(m.bias.data, 0.0)
        #     elif classname.find(
        #             'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        #         init.normal_(m.weight.data, 1.0, init_gain)
        #         init.constant_(m.bias.data, 0.0)
        #
        # print('initialize network with %s' % init_type)
        # net.apply(init_func)
