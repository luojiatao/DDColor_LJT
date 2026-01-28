import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import functools
from torch.optim import lr_scheduler
from fastai.vision.all import *
# from models.mobile_resnet_generator import MobileResnetGenerator


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=opt.lr * 0.05)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_net_exclude(net, init_type='normal', init_gain=0.02, gpu_ids=[], exclude_layers=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    for i in range(len(net.layers)):
        if exclude_layers.count(i) <= 0:
            init_weights(net[i], init_type, init_gain=init_gain)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'resunet':
        arch = resnet34
        img_size = torch.Size([320, 320])
        net = create_unet_model(arch, output_nc, img_size, pretrained=True, blur=True,
                                norm_type=NormType.Weight, self_attention=False)
        # 不初始化backbone部分
        return init_net_exclude(net, init_type, init_gain, gpu_ids, exclude_layers=[0])
    elif netG == 'resunet2':  # 主要学残差
        arch = resnet34
        img_size = torch.Size([320, 320])
        net = ResUnet(arch, output_nc, img_size, pretrained=True, blur=True,
                      norm_type=NormType.Weight, self_attention=False)
        # 不初始化backbone部分
        return init_net_exclude(net, init_type, init_gain, gpu_ids, exclude_layers=[0])
    elif netG == 'resunet4c':  # 四通道输入
        arch = resnet34
        img_size = torch.Size([320, 320])
        net = ResUnet4C(arch, output_nc, img_size, pretrained=True, blur=True,
                        norm_type=NormType.Weight, self_attention=False)
        # 不初始化backbone部分
        return init_net_exclude(net, init_type, init_gain, gpu_ids, exclude_layers=[0])
    elif netG == 'resunet4c_noblur':  # 四通道输入
        arch = resnet34
        img_size = torch.Size([320, 320])
        net = ResUnet4C(arch, output_nc, img_size, pretrained=True, blur=False,
                        norm_type=NormType.Weight, self_attention=False)
        # 不初始化backbone部分
        return init_net_exclude(net, init_type, init_gain, gpu_ids, exclude_layers=[0])
    elif netG == 'resunet6c':  # 四通道输入
        arch = resnet34
        img_size = torch.Size([320, 320])
        net = ResUnet6C(arch, output_nc, img_size, pretrained=True, blur=True,
                        norm_type=NormType.Weight, self_attention=False)
        # 不初始化backbone部分
        return init_net_exclude(net, init_type, init_gain, gpu_ids, exclude_layers=[0])
    elif netG == 'rrdbnet':
        net = RRDBNet(input_nc, output_nc, scale=1, num_feat=64, num_block=23, num_grow_ch=32)
    # elif netG == 'mobile_resnet_9blocks':
    #     net = MobileResnetGenerator(input_nc, output_nc, ngf=ngf, norm_layer=norm_layer,
    #                                 dropout_rate=0, n_blocks=9)
    # elif netG == 'mobile_deepest_resnet':
    #     from models.mobile_resnet_deepest_generator import MobileResnetDeepestGenerator
    #     net = MobileResnetDeepestGenerator(input_nc, output_nc, ngf=ngf, norm_layer=norm_layer,
    #                                        dropout_rate=0, n_blocks=9)
    # elif netG == 'retouchingnet':
    #     net = RetouchingNet(input_nc, output_nc)
    # elif netG == 'retouchingstylenet':
    #     net = RetouchingStyleNet(input_nc, output_nc)
    # elif netG == 'retouchingnetlite':
    #     from models.networkslite import RetouchingNetLite
    #     net = RetouchingNetLite(input_nc, output_nc)
    # elif netG == 'fanet':
    #     from models.networkslite import FANet
    #     net = FANet(input_nc, output_nc)
    # elif netG == 'fanetlite':
    #     from models.networkslite import FANetlite
    #     net = FANetlite(input_nc, output_nc)
    # elif netG == 'tjhfaceretouchnet':
    #     from models.faceretouchnet import TJHFaceRetouchNet
    #     net = TJHFaceRetouchNet(input_nc, output_nc, ngf=ngf)
    # elif netG == 'tjhfaceretouchnet_deep':
    #     from models.faceretouchnet import TJHFaceRetouchNet
    #     net = TJHFaceRetouchNet(input_nc, output_nc, ngf=ngf, expand_deep=True)
    # elif netG == 'gcfsr':
    #     from models.gcfsr_arch import GCFSR_blind
    #     net = GCFSR_blind(512)
    #     # FIXME init有点问题先跳过
    #     if len(gpu_ids) > 0:
    #         assert (torch.cuda.is_available())
    #         net.to(gpu_ids[0])
    #         net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    #     return net
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options (extend PatchGAN)
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'multiscale_animesr':
        from models.discriminators_animesr import MultiScaleDiscriminator
        # FIXME
        net = MultiScaleDiscriminator(num_in_ch=input_nc,
                                      num_feat=ndf,
                                      num_layers=3,
                                      max_nf_mult=8,
                                      norm_type='none',
                                      use_sigmoid=False,
                                      use_sn=True,
                                      use_downscale=True)
    elif netD == 'basicsr_unetdiscriminatorsn':
        from models.discriminators_basicsr import UNetDiscriminatorSN
        net = UNetDiscriminatorSN(num_in_ch=input_nc)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'wgan_softplus']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'wgan_softplus':
            if target_is_real:
                loss = F.softplus(-prediction).mean()
            else:
                loss = F.softplus(prediction).mean()

        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


# TODO
# class MultiNLayerDiscriminator(BaseNetwork):
#     """Defines a PatchGAN discriminator"""
#
#     def __init__(self, input_nc, n_share, ndf=64, norm_layer=nn.BatchNorm2d):
#         """Construct a PatchGAN discriminator
#
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         self.n_share = n_share
#         super(MultiNLayerDiscriminator, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         kw = 4
#         padw = 1
#
#         block1s = []
#         block2s = []
#         block3s = []
#         block4s = []
#         block5s = []
#
#         for _ in [0, 1]:
#             block1s.append(ConvReLU(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
#             block2s.append(ConvBNReLU(ndf * 1, ndf * 2, kernel_size=kw, stride=2, padding=padw, norm_layer=norm_layer, use_bias=use_bias))
#             block3s.append(ConvBNReLU(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, norm_layer=norm_layer, use_bias=use_bias))
#             block4s.append(ConvBNReLU(ndf * 4, ndf * 8, kernel_size=kw, stride=1, padding=padw, norm_layer=norm_layer, use_bias=use_bias))
#             block5s.append(Conv(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw))
#
#         self.block1s = nn.ModuleList(block1s)
#         self.block2s = nn.ModuleList(block2s)
#         self.block3s = nn.ModuleList(block3s)
#         self.block4s = nn.ModuleList(block4s)
#         self.block5s = nn.ModuleList(block5s)
#
#     def forward(self, input):
#         idx = 0 if FLAGS.teacher_ids == 1 else 1
#
#         h = input
#         h = self.block1s[-1 if self.n_share > 0 else idx](h)
#         h = self.block2s[-1 if self.n_share > 1 else idx](h)
#         h = self.block3s[-1 if self.n_share > 2 else idx](h)
#         h = self.block4s[-1 if self.n_share > 3 else idx](h)
#         output = self.block5s[-1 if self.n_share > 4 else idx](h)
#         return output

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


################################################
class ResUnet(nn.Module):
    def __init__(self, arch, n_out, img_size, pretrained=True, **kwargs):
        super(ResUnet, self).__init__()
        self.alpha = 0.6
        self.unet = create_unet_model(arch, n_out, img_size, pretrained=pretrained, **kwargs)
        self.layers = self.unet.layers

    def forward(self, x):
        res = x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig, nres.orig = None, None
            res = nres
        return x * self.alpha + res * (1. - self.alpha)

    def __getitem__(self, i): return self.layers[i]  # subscriptable
    # def append(self,l):      return self.layers.append(l)
    # def extend(self,l):      return self.layers.extend(l)
    # def insert(self,i,l):    return self.layers.insert(i,l)


class ResUnet4C(nn.Module):
    def __init__(self, arch, n_out, img_size, pretrained=True, **kwargs):
        super(ResUnet4C, self).__init__()
        self.alpha = 0.6
        self.unet = create_unet_model(arch, n_out, img_size, pretrained=pretrained, **kwargs)
        self.layers = self.unet.layers
        print(self.layers[0][0])
        # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # fixme
        weight = self.layers[0][0].weight.clone()
        self.layers[0][0] = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init_weights(self.layers[0][0], init_type='kaiming', init_gain=0.02)

        with torch.no_grad():
            self.layers[0][0].weight[:, :3] = weight

        print(self.layers[0][0])

        # print(self.layers[11])
        self.layers[11].convpath[0][0] = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.layers[11].convpath[1][0] = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)

        # print(self.layers[12])
        self.layers[12][0] = nn.Conv2d(100, 3, kernel_size=1, stride=1)

        # self.layers[11].idpath = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        # exit(0)

    def forward(self, x):
        res = x
        idx = 0
        for l in self.layers:
            # print(idx)
            # if idx == 12:
            #     print(res.shape)
            #     print(l)
            res.orig = x
            nres = l(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig, nres.orig = None, None
            res = nres
            # idx += 1
        # return x[:,:3,:,:] * self.alpha + res * (1. - self.alpha)
        return res

    def __getitem__(self, i): return self.layers[i]  # subscriptable
    # def append(self,l):      return self.layers.append(l)
    # def extend(self,l):      return self.layers.extend(l)
    # def insert(self,i,l):    return self.layers.insert(i,l)


class ResUnet6C(nn.Module):
    def __init__(self, arch, n_out, img_size, pretrained=True, **kwargs):
        super(ResUnet6C, self).__init__()
        self.alpha = 0.6
        self.unet = create_unet_model(arch, n_out, img_size, pretrained=pretrained, **kwargs)
        self.layers = self.unet.layers
        print(self.layers[0][0])
        # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # fixme
        weight = self.layers[0][0].weight.clone()
        self.layers[0][0] = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init_weights(self.layers[0][0], init_type='kaiming', init_gain=0.02)

        with torch.no_grad():
            self.layers[0][0].weight[:, :3] = weight

        print(self.layers[0][0])

        # print(self.layers[11])
        self.layers[11].convpath[0][0] = nn.Conv2d(102, 102, kernel_size=3, stride=1, padding=1)
        self.layers[11].convpath[1][0] = nn.Conv2d(102, 102, kernel_size=3, stride=1, padding=1)

        # print(self.layers[12])
        self.layers[12][0] = nn.Conv2d(102, n_out, kernel_size=1, stride=1)

        # self.layers[11].idpath = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        # exit(0)

    def forward(self, x):
        res = x
        idx = 0
        for l in self.layers:
            # print(idx)
            # if idx == 12:
            #     print(res.shape)
            #     print(l)
            res.orig = x
            nres = l(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig, nres.orig = None, None
            res = nres
            # idx += 1
        # return x[:,:3,:,:] * self.alpha + res * (1. - self.alpha)
        return res

    def __getitem__(self, i): return self.layers[i]  # subscriptable
    # def append(self,l):      return self.layers.append(l)
    # def extend(self,l):      return self.layers.extend(l)
    # def insert(self,i,l):    return self.layers.insert(i,l)


################################################
import torchvision.models.vgg as vgg
# 延迟导入 mmcv，避免在不需要 PerceptualVGG 时强制依赖
# from mmcv.runner import load_checkpoint


class PerceptualVGG(nn.Module):
    """VGG network used in calculating perceptual loss.
    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.
    Args:
        layer_name_list (list[str]): According to the name in this list,
            forward function will return the corresponding features. This
            list contains the name each layer in `vgg.feature`. An example
            of this list is ['4', '10'].
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image.
            Importantly, the input feature must in the range [0, 1].
            Default: True.
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 pretrained='torchvision://vgg19'):
        super().__init__()
        if pretrained.startswith('torchvision://'):
            assert vgg_type in pretrained
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm

        # get vgg model and load pretrained vgg weight
        # remove _vgg from attributes to avoid `find_unused_parameters` bug
        _vgg = getattr(vgg, vgg_type)()
        self.init_weights(_vgg, pretrained)
        num_layers = max(map(int, layer_name_list)) + 1
        assert len(_vgg.features) >= num_layers
        # only borrow layers that will be used from _vgg to avoid unused params
        # https://blog.csdn.net/susansmile1014/article/details/77216651
        # Sequential(
        #   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (1): ReLU(inplace=True)
        #   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (3): ReLU(inplace=True)
        #   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        #   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (6): ReLU(inplace=True)
        #   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (8): ReLU(inplace=True)
        #   (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        #   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (11): ReLU(inplace=True)
        #   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (13): ReLU(inplace=True)
        #   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (15): ReLU(inplace=True)
        #   (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (17): ReLU(inplace=True)
        #   (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        #   (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (20): ReLU(inplace=True)
        #   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (22): ReLU(inplace=True)
        #   (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (24): ReLU(inplace=True)
        #   (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (26): ReLU(inplace=True)
        #   (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        #   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (29): ReLU(inplace=True)
        #   (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (31): ReLU(inplace=True)
        #   (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (33): ReLU(inplace=True)
        #   (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (35): ReLU(inplace=True)
        #   (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # )
        # print(_vgg.features)
        # exit(0)
        self.vgg_layers = _vgg.features[:num_layers]

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [-1, 1]
            self.register_buffer(
                'std',
                torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for v in self.vgg_layers.parameters():
            v.requires_grad = False

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}

        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_list:
                output[name] = x.clone()
        return output

    def init_weights(self, model, pretrained):
        """Init weights.
        Args:
            model (nn.Module): Models to be inited.
            pretrained (str): Path for pretrained weights.
        """
        # logger = get_root_logger()
        # print('vgg', model)
        from mmcv.runner import load_checkpoint  # 延迟导入
        load_checkpoint(model, pretrained, logger=None)


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.
    Args:
        layers_weights (dict): The weight for each layer of vgg feature for
            perceptual loss. Here is an example: {'4': 1., '9': 1., '18': 1.},
            which means the 5th, 10th and 18th feature layer will be
            extracted with weight 1.0 in calculating losses.
        layers_weights_style (dict): The weight for each layer of vgg feature
            for style loss. If set to 'None', the weights are set equal to
            the weights for perceptual loss. Default: None.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 1.0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward function of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'.
        criterion (str): Criterion type. Options are 'l1' and 'mse'.
            Default: 'l1'.
    """

    def __init__(self,
                 layer_weights={'4': 1., '9': 1., '18': 1.},
                 layer_weights_style=None,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 pretrained='torchvision://vgg19',
                 criterion='l1',
                 weight_p=1.0,
                 weight_s=1.0, ):
        super().__init__()
        self.layer_weights = layer_weights
        self.layer_weights_style = layer_weights_style
        self.weight_p = weight_p
        self.weight_s = weight_s

        self.vgg = PerceptualVGG(
            layer_name_list=list(self.layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            pretrained=pretrained)

        if self.layer_weights_style is not None and \
                self.layer_weights_style != self.layer_weights:
            self.vgg_style = PerceptualVGG(
                layer_name_list=list(self.layer_weights_style.keys()),
                vgg_type=vgg_type,
                use_input_norm=use_input_norm,
                pretrained=pretrained)
        else:
            self.layer_weights_style = self.layer_weights
            self.vgg_style = None

        criterion = criterion.lower()
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).           [0, 1]
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).   [0, 1]
        Returns:
            Tensor: Forward results.
        """

        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        percep_loss = 0
        if self.weight_p > 0:
            for k in x_features.keys():
                percep_loss += self.criterion(
                    x_features[k], gt_features[k]) * self.layer_weights[k]

        # calculate style loss
        style_loss = 0
        if self.weight_s > 0:
            if self.vgg_style is not None:
                x_features = self.vgg_style(x)
                gt_features = self.vgg_style(gt.detach())

            for k in x_features.keys():
                style_loss += self.criterion(
                    self._gram_mat(x_features[k]),
                    self._gram_mat(
                        gt_features[k])) * self.layer_weights_style[k]

        return (self.weight_p * percep_loss, self.weight_s * style_loss), (percep_loss, style_loss)

    def _gram_mat(self, x):
        """Calculate Gram matrix.
        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).
        Returns:
            torch.Tensor: Gram matrix.
        """
        (n, c, h, w) = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


##########################################################
class TotalVariationLoss(nn.Module):
    def __init__(self, k_size):
        super().__init__()
        self.k_size = k_size

    def forward(self, image: torch.Tensor):
        b, c, h, w = image.shape
        tv_h = torch.mean((image[:, :, self.k_size:, :] - image[:, :, : -self.k_size, :]) ** 2)
        tv_w = torch.mean((image[:, :, :, self.k_size:] - image[:, :, :, : -self.k_size]) ** 2)
        tv_loss = (tv_h + tv_w) / (3 * h * w)
        return tv_loss


##########################################################
from torch.nn.modules.batchnorm import _BatchNorm


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.
    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.
    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.
    Used in RRDB block in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.
    Used in RRDB-Net in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.num_in_ch = num_in_ch
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        # FIXME 额外加的最后一层
        # haircolor_resunet2_exp2403071000
        self.last_conv = nn.Conv2d(in_channels=self.num_in_ch + num_out_ch, out_channels=num_out_ch, kernel_size=1, stride=1,
                                   padding=0)
        # self.last_conv = nn.Conv2d(in_channels=num_out_ch + num_out_ch, out_channels=num_out_ch, kernel_size=1,
        #                            stride=1,
        #                            padding=0)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        # print(torch.cat((out, x), dim=1).shape)
        # haircolor_resunet2_exp2403071000
        out = self.last_conv(torch.cat((out, x), dim=1))

        # out = self.last_conv(torch.cat((out, x[:, :3, :, :]), dim=1))
        return out


##########################################################
# from modelscope/modelscope/models/cv/skin_retouching/inpainting_model/gconv.py
class GatedConvBNActiv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bn=True,
                 sample='none-3',
                 activ='relu',
                 bias=False):
        super(GatedConvBNActiv, self).__init__()

        if sample == 'down-7':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=bias)
        elif sample == 'down-5':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=bias)
        elif sample == 'down-3':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        images = self.conv(x)
        gates = self.sigmoid(self.gate(x))

        if hasattr(self, 'bn'):
            images = self.bn(images)
        if hasattr(self, 'activation'):
            images = self.activation(images)

        images = images * gates

        return images


class GatedConvBNActiv2(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bn=True,
                 sample='none-3',
                 activ='relu',
                 bias=False):
        super(GatedConvBNActiv2, self).__init__()

        if sample == 'down-7':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=bias)
        elif sample == 'down-5':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=bias)
        elif sample == 'down-3':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)

        self.conv_skip = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, f_up, f_skip, mask):
        x = torch.cat((f_up, f_skip, mask), dim=1)
        images = self.conv(x)
        images_skip = self.conv_skip(f_skip)
        gates = self.sigmoid(self.gate(x))

        if hasattr(self, 'bn'):
            images = self.bn(images)
            images_skip = self.bn(images_skip)
        if hasattr(self, 'activation'):
            images = self.activation(images)
            images_skip = self.activation(images_skip)

        images = images * gates + images_skip * (1 - gates)

        return images


# https://github.com/modelscope/modelscope/blob/db0f70bc1c9e1c94448d5a91b757c44c5bb685f5/modelscope/pipelines/cv/skin_retouching_pipeline.py#L14
class RetouchingNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 up_sampling_node='nearest',
                 need_mask=False):
        super(RetouchingNet, self).__init__()

        self.freeze_ec_bn = False
        self.up_sampling_node = up_sampling_node

        self.ec_images_1 = GatedConvBNActiv(
            in_channels, 64, bn=False, sample='down-3')
        self.ec_images_2 = GatedConvBNActiv(64, 128, sample='down-3')
        self.ec_images_3 = GatedConvBNActiv(128, 256, sample='down-3')
        self.ec_images_4 = GatedConvBNActiv(256, 512, sample='down-3')
        self.ec_images_5 = GatedConvBNActiv(512, 512, sample='down-3')
        self.ec_images_6 = GatedConvBNActiv(512, 512, sample='down-3')

        self.dc_images_6 = GatedConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_images_5 = GatedConvBNActiv(512 + 512, 512, activ='leaky')  # 14.747% MACs
        self.dc_images_4 = GatedConvBNActiv(512 + 256, 256, activ='leaky')  # 22.122% MACs
        self.dc_images_3 = GatedConvBNActiv(256 + 128, 128, activ='leaky')  # 22.127% MACs
        self.dc_images_2 = GatedConvBNActiv(128 + 64, 64, activ='leaky')  # 22.136% MACs,
        self.dc_images_1 = GatedConvBNActiv(
            64 + in_channels,
            out_channels,
            bn=False,
            sample='none-3',
            activ=None,
            bias=True)

        self.tanh = nn.Tanh()
        self.need_mask = need_mask

    def forward(self, input_images, input_masks=None):
        # print('input_images.shape', input_images.shape)

        ec_images = {}

        if self.need_mask:
            ec_images['ec_images_0'] = torch.cat((input_images, input_masks),
                                                 dim=1)
        else:
            ec_images['ec_images_0'] = input_images

        ec_images['ec_images_1'] = self.ec_images_1(ec_images['ec_images_0'])
        ec_images['ec_images_2'] = self.ec_images_2(ec_images['ec_images_1'])
        ec_images['ec_images_3'] = self.ec_images_3(ec_images['ec_images_2'])

        ec_images['ec_images_4'] = self.ec_images_4(ec_images['ec_images_3'])
        ec_images['ec_images_5'] = self.ec_images_5(ec_images['ec_images_4'])
        ec_images['ec_images_6'] = self.ec_images_6(ec_images['ec_images_5'])

        # --------------
        # images decoder
        # --------------
        dc_images = ec_images['ec_images_6']
        for _ in range(6, 0, -1):
            ec_images_skip = 'ec_images_{:d}'.format(_ - 1)
            dc_conv = 'dc_images_{:d}'.format(_)

            dc_images = F.interpolate(
                dc_images, scale_factor=2, mode=self.up_sampling_node)

            # print('dc_images.shape', dc_images.shape)
            # print('ec_images[ec_images_skip].shape', ec_images[ec_images_skip].shape)

            dc_images = torch.cat((dc_images, ec_images[ec_images_skip]),
                                  dim=1)

            dc_images = getattr(self, dc_conv)(dc_images)

        outputs = dc_images  # (self.tanh(dc_images) + 1.) / 2.

        return outputs


# from https://github.com/lucidrains/stylegan2-pytorch/blob/05f7585e8da9c09752696872e04de0414a972486/stylegan2_pytorch/stylegan2_pytorch.py
class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps=1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, z_id):
        b, c, h, w = x.shape

        w1 = z_id[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, dilation={self.dilation}, demodulation={self.demod})"
        )


def set_activate_layer(types):
    # initialize activation
    if types == 'relu':
        activation = nn.ReLU()
    elif types == 'relu6':
        activation = nn.ReLU6()
    elif types == 'lrelu':
        activation = nn.LeakyReLU(0.2, inplace=True)
    elif types == 'mish':
        activation = nn.Mish(inplace=True)
    elif types == "swish":
        activation = nn.SiLU(inplace=True)
    elif types == 'tanh':
        activation = nn.Tanh()
    elif types == 'sig':
        activation = nn.Sigmoid()
    elif types == 'none':
        activation = nn.Identity()
    else:
        assert 0, f"Unsupported activation: {types}"
    return activation


class GenResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=256,
                 activation='lrelu', up_sample=False):
        super().__init__()
        self.actv = set_activate_layer(activation)
        if up_sample:
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up_sample = nn.Identity()
        self.conv1 = Conv2DMod(dim_in, dim_out, 3, stride=1, dilation=1)
        self.conv2 = Conv2DMod(dim_out, dim_out, 3, stride=1, dilation=1)
        self.style1 = nn.Linear(style_dim, dim_in)
        self.style2 = nn.Linear(style_dim, dim_out)
        if dim_in != dim_out:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        else:
            self.conv1x1 = nn.Identity()

    def forward(self, x, s):
        x = self.up_sample(x)
        x_ = self.conv1x1(x)
        s1 = self.style1(s)
        x = self.conv1(x, s1)
        x = self.actv(x)
        s2 = self.style2(s)
        x = self.conv2(x, s2)
        x = self.actv(x + x_)
        return x


class GatedConvINActiv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm=True,
                 sample='none-3',
                 activ='relu',
                 bias=False):
        super(GatedConvINActiv, self).__init__()

        if sample == 'down-7':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=bias)
        elif sample == 'down-5':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=bias)
        elif sample == 'down-3':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)

        if norm:
            self.norm = nn.InstanceNorm2d(out_channels)

        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        images = self.conv(x)
        gates = self.sigmoid(self.gate(x))

        if hasattr(self, 'norm'):
            images = self.norm(images)
        if hasattr(self, 'activation'):
            images = self.activation(images)

        images = images * gates

        return images


class GatedConvINActiv2(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm=True,
                 sample='none-3',
                 activ='relu',
                 bias=False):
        super(GatedConvINActiv2, self).__init__()

        if sample == 'down-7':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=bias)
        elif sample == 'down-5':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=bias)
        elif sample == 'down-3':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)

        self.conv_skip = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias)

        if norm:
            self.norm = nn.InstanceNorm2d(out_channels)

        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, f_up, f_skip, mask):
        x = torch.cat((f_up, f_skip, mask), dim=1)
        images = self.conv(x)
        images_skip = self.conv_skip(f_skip)
        gates = self.sigmoid(self.gate(x))

        if hasattr(self, 'norm'):
            images = self.norm(images)
            images_skip = self.norm(images_skip)
        if hasattr(self, 'activation'):
            images = self.activation(images)
            images_skip = self.activation(images_skip)

        images = images * gates + images_skip * (1 - gates)

        return images


class ECA(nn.Module):
    def __init__(self, channel, beta=1, gamma=2):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + beta) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, down_sample=False, up_sample=False, bn=False, attention=False,
                 activation='lrelu'):
        super(ResBlock, self).__init__()
        main_module_list = []
        if bn:
            main_module_list += [
                nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False),
                nn.BatchNorm2d(in_channel)
            ]
        else:
            main_module_list += [
                nn.InstanceNorm2d(in_channel, affine=True),
                set_activate_layer(activation),
                nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False),
            ]
        if down_sample:
            main_module_list.append(nn.AvgPool2d(kernel_size=2))
        elif up_sample:
            main_module_list += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ]
        if bn:
            main_module_list += [
                nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channel)
            ]
        else:
            main_module_list += [
                nn.InstanceNorm2d(in_channel, affine=True),
                set_activate_layer(activation),
                nn.Conv2d(in_channel, out_channel, 3, 1, 1)
            ]
        if attention:
            main_module_list += [
                ECA(out_channel)
            ]
        self.main_path = nn.Sequential(*main_module_list)
        side_module_list = []
        if in_channel != out_channel:
            side_module_list += [nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)]
        else:
            side_module_list += [nn.Identity()]
        if down_sample:
            side_module_list.append(nn.AvgPool2d(kernel_size=2))
        elif up_sample:
            side_module_list += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ]
        self.side_path = nn.Sequential(*side_module_list)

    def forward(self, x):
        x1 = self.main_path(x)
        x2 = self.side_path(x)
        return (x1 + x2) / math.sqrt(2)


class RetouchingStyleNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 style_dim=128,
                 up_sampling_node='bilinear',
                 need_mask=False):
        super(RetouchingStyleNet, self).__init__()

        self.up_sampling_node = up_sampling_node

        self.style_dim = style_dim
        # self.style_id = nn.Parameter(torch.randn((self.style_dim)))

        self.first = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.ec_images_1 = ResBlock(64, 64, down_sample=True, attention=True, activation='lrelu')
        self.ec_images_2 = ResBlock(64, 128, down_sample=True, attention=False, activation='lrelu')
        self.ec_images_3 = ResBlock(128, 256, down_sample=True, attention=True, activation='lrelu')
        self.ec_images_4 = ResBlock(256, 512, down_sample=True, attention=False, activation='lrelu')
        self.ec_images_5 = ResBlock(512, 512, down_sample=True, attention=True, activation='lrelu')
        self.ec_images_6 = ResBlock(512, 512, down_sample=True, attention=False, activation='lrelu')

        self.dc_images_6 = GatedConvINActiv(512 + 512, 512, activ='leaky')
        self.dc_images_5 = GatedConvINActiv(512 + 512, 512, activ='leaky')
        self.dc_images_4 = GatedConvINActiv(512 + 256, 256, activ='leaky')
        self.dc_images_3 = GatedConvINActiv(256 + 128, 128, activ='leaky')
        self.dc_images_2 = GatedConvINActiv(128 + 64, 64, activ='leaky')
        self.dc_images_1 = GatedConvINActiv(
            64 + in_channels,
            out_channels,
            norm=False,
            sample='none-3',
            activ=None,
            bias=True)

        # self.apply_style_6 = GenResBlk(dim_in=512, dim_out=512, style_dim=self.style_dim, activation='lrelu',
        #                                up_sample=True)
        # self.apply_style_5 = GenResBlk(dim_in=512, dim_out=512, style_dim=self.style_dim, activation='lrelu',
        #                                up_sample=True)
        # self.apply_style_4 = GenResBlk(dim_in=512, dim_out=512, style_dim=self.style_dim, activation='lrelu',
        #                                up_sample=True)
        # self.apply_style_3 = GenResBlk(dim_in=256, dim_out=256, style_dim=self.style_dim, activation='lrelu',
        #                                up_sample=True)
        # self.apply_style_2 = GenResBlk(dim_in=128, dim_out=128, style_dim=self.style_dim, activation='lrelu',
        #                                up_sample=True)
        # self.apply_style_1 = GenResBlk(dim_in=64, dim_out=64, style_dim=self.style_dim, activation='lrelu',
        #                                up_sample=True)

        self.need_mask = need_mask

    def forward(self, input_images, input_masks=None):
        ec_images = {}

        if self.need_mask:
            ec_images['ec_images_0'] = torch.cat((input_images, input_masks),
                                                 dim=1)
        else:
            ec_images['ec_images_0'] = input_images

        ec_images['ec_images_1'] = self.ec_images_1(self.first(ec_images['ec_images_0']))

        ec_images['ec_images_2'] = self.ec_images_2(ec_images['ec_images_1'])
        ec_images['ec_images_3'] = self.ec_images_3(ec_images['ec_images_2'])

        ec_images['ec_images_4'] = self.ec_images_4(ec_images['ec_images_3'])
        ec_images['ec_images_5'] = self.ec_images_5(ec_images['ec_images_4'])
        ec_images['ec_images_6'] = self.ec_images_6(ec_images['ec_images_5'])

        # --------------
        # images decoder
        # --------------
        dc_images = ec_images['ec_images_6']

        # expanded_style_id = self.style_id.unsqueeze(0).expand(input_images.shape[0], -1)
        for _ in range(6, 0, -1):
            ec_images_skip = 'ec_images_{:d}'.format(_ - 1)
            dc_conv = 'dc_images_{:d}'.format(_)

            dc_images = F.interpolate(
                dc_images, scale_factor=2, mode=self.up_sampling_node)
            # print('dc_images.shape',dc_images.shape)

            # apply_style_blk = 'apply_style_{:d}'.format(_)
            # dc_images = getattr(self, apply_style_blk)(dc_images, expanded_style_id)
            # print('dc_images.shape', dc_images.shape)
            # print('ec_images[ec_images_skip].shape', ec_images[ec_images_skip].shape)

            dc_images = torch.cat((dc_images, ec_images[ec_images_skip]),
                                  dim=1)
            dc_images = getattr(self, dc_conv)(dc_images)

        outputs = dc_images

        return outputs


if __name__ == '__main__':
    from thop import profile
    from thop import clever_format

    from ptflops import get_model_complexity_info

    net = RetouchingNet()
    net.eval().cuda()
    input = torch.rand(1, 3, 512, 512).cuda()
    # dummy_mask = torch.rand(1, 1, 512, 512).cuda()
    # out = net(input)
    # print('out.shape', out.shape)

    macs, params = profile(net, inputs=(input,))

    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)

    macs, params = get_model_complexity_info(net, (3, 512, 512), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    pass
