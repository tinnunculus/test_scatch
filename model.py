import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from math import sqrt

def _XDoG(img, gamma=0.98, phi=200, epsilon=0.1, kappa=1.6, sigma=0.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gFiltered1 = cv2.GaussianBlur(gray, (11, 11), sigma)
    gFiltered2 = cv2.GaussianBlur(gray, (11, 11), kappa*sigma)
    diff = gFiltered1 - (gamma*gFiltered2)
    xdog = np.zeros_like(diff)
    mask = diff > epsilon
    mask_neg = np.logical_not(mask)
    xdog[mask] = 1.
    xdog[mask_neg] = 1. + np.tanh(phi*(diff[mask_neg]-epsilon))
    return xdog

def XDoG_threshold(img, gamma=0.98, phi=200, epsilon=0.00001, kappa=1.6, sigma=0.8, thr=1., use_blurry_input=True):
    if use_blurry_input:
        img = cv2.GaussianBlur(img, (11, 11), 0.7).astype(np.float32)
    xdog = _XDoG(img, gamma, phi, epsilon, kappa, sigma)
    return xdog.astype(np.float32)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size = 3, stride = 1)
        self.conv2 = ConvLayer(channels, channels, kernel_size = 3, stride = 1)
        self.relu = torch.nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu((self.conv1(x)))
        out = self.conv2(out)
        out = out * 0.1
        out = out + residual
        return out

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size = 128):
        return input.view(input.size(0), size, 1, 1)

class real_scatch_ref_encoder(nn.Module):
    def __init__(self, image_size = 256, z_dim = 128):
        super(real_scatch_ref_encoder, self).__init__()

        self.encoder = nn.Sequential(
            ConvLayer(3, 128, kernel_size = 5, stride = 1),
            nn.PReLU(),
            ConvLayer(128, 128, kernel_size = 3, stride = 2),
            nn.PReLU(),
            ConvLayer(128, 128, kernel_size = 3, stride = 2),
            nn.PReLU(),
            ConvLayer(128, 128, kernel_size = 3, stride = 2),
            nn.PReLU(),
            ResidualBlock(128),
            Flatten()
        )

        self.fc1_1 = nn.Linear((int)(image_size / 8) * (int)(image_size / 8) * 128, z_dim)
        self.fc2_1 = nn.Linear((int)(image_size / 8) * (int)(image_size / 8) * 128, z_dim)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(torch.device("cuda"))
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu = self.fc1_1(h)
        logvar = self.fc2_1(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

class real_scatch_ref_decoder(nn.Module):
    def __init__(self, image_size = 256, z_dim = 128):
        super(real_scatch_ref_decoder, self).__init__()
        self.image_size = image_size
        self.z_dim = z_dim
        self.Upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 128,kernel_size = 4, stride = 2, padding = 1),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 128,kernel_size = 4, stride = 2, padding = 1),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 128,kernel_size = 4, stride = 2, padding = 1),
            nn.PReLU(),
            ConvLayer(128, 128, kernel_size = 3, stride=1),
            nn.PReLU(),
            ConvLayer(128, 128, kernel_size = 3, stride=1),
            nn.PReLU(),
            ConvLayer(128, 128, kernel_size = 3, stride=1),
            nn.PReLU(),
            ConvLayer(128, 128, kernel_size = 3, stride=1),
            nn.PReLU(),
            ConvLayer(128, 3, kernel_size = 3, stride=1),
            nn.PReLU(),
            nn.Sigmoid()
        )

        self.fc3 = nn.Linear(z_dim, z_dim * 2)
        self.fc4 = nn.Linear(z_dim * 2, (int)(image_size / 8) * (int)(image_size / 8) * 128)

    def forward(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        input = z.view(z.size(0), 128, (int)(self.image_size / 8), (int)(self.image_size / 8))
        out = self.Upsample(input)
        return out

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels,out_channels, up_scale = 2):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise

class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None

class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None

blur = BlurFunction.apply

class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, latent=128):
        super().__init__()

        self.conv1 = nn.Sequential(
            FusedUpsample(in_channel, out_channel, kernel_size, padding=padding),
            Blur(out_channel)
        )
        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.lrelu = nn.LeakyReLU(0.2)
        self.ca_layer_1 = CALayer(out_channel)

        self.conv2 = ConvLayer(out_channel, out_channel, kernel_size, stride=1)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, latent)
        self.ca_layer_2 = CALayer(out_channel)

        self.conv3 = ConvLayer(out_channel, out_channel, kernel_size, stride=1)
        self.noise3 = equal_lr(NoiseInjection(out_channel))
        self.ca_layer_3 = CALayer(out_channel)

        self.conv4 = ConvLayer(out_channel, out_channel, kernel_size, stride=1)
        self.noise4 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, latent)
        self.ca_layer_4 = CALayer(out_channel)

        self.conv5 = ConvLayer(out_channel, out_channel, kernel_size, stride=1)
        self.noise5 = equal_lr(NoiseInjection(out_channel))
        self.ca_layer_5 = CALayer(out_channel)

    def forward(self, input, style):
        batch = input.size(0)
        noise_1 = torch.randn(batch, 1, input.size(2) * 2, input.size(3)*2, device=input[0].device)
        noise_2 = torch.randn(batch, 1, input.size(2) * 2, input.size(3)*2, device=input[0].device)
        noise_3 = torch.randn(batch, 1, input.size(2) * 2, input.size(3)*2, device=input[0].device)
        noise_4 = torch.randn(batch, 1, input.size(2) * 2, input.size(3)*2, device=input[0].device)
        noise_5 = torch.randn(batch, 1, input.size(2) * 2, input.size(3)*2, device=input[0].device)

        out_1 = self.conv1(input)
        out = self.noise1(out_1, noise_1)
        out = self.lrelu(out)
        out = self.ca_layer_1(out)

        out = self.conv2(out)
        out = self.noise2(out, noise_2)
        out = self.lrelu(out)
        out = self.ca_layer_3(out)
        out = self.adain1(out, style)

        out = self.conv3(out)
        out = self.noise3(out, noise_3)
        out = self.lrelu(out)
        out = self.ca_layer_3(out)

        out = self.conv4(out)
        out = self.noise4(out,noise_4)
        out = self.lrelu(out)
        out = self.ca_layer_4(out)
        out = self.adain2(out, style)

        out = self.conv5(out)
        out = self.noise5(out, noise_5)
        out = self.lrelu(out)
        out = self.ca_layer_5(out)

        return out + out_1

class generator(nn.Module):
    def __init__(self, input_size = 512, latent_dim = 128):
        super(generator,self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.initial = ConvLayer(in_channels=3, out_channels=256, kernel_size=3, stride=1)
        self.upsample_1 = BasicBlock(256, 256, latent=self.latent_dim)

        self.origin_2 = ConvLayer(in_channels=3, out_channels=256, kernel_size=3, stride=1)
        self.upsample_2 = BasicBlock(256, 256, latent=self.latent_dim)

        self.origin_3 = ConvLayer(in_channels=3, out_channels=256, kernel_size=3, stride=1)
        self.upsample_3 = BasicBlock(256, 128, latent=self.latent_dim)

        self.origin_4 = ConvLayer(in_channels=3, out_channels=128, kernel_size=3, stride=1)
        self.upsample_4 = BasicBlock(128, 128, latent=self.latent_dim)

        self.origin_5 = ConvLayer(in_channels=3, out_channels=128, kernel_size=3, stride=1)
        self.upsample_5 = BasicBlock(128, 64, latent=self.latent_dim)

        self.origin_6 = ConvLayer(in_channels=3, out_channels=64, kernel_size=3, stride=1)
        self.upsample_6 = BasicBlock(64, 32, latent=self.latent_dim)

        self.upsample_7 = UpsampleBLock(32, 32, 2)

        self.finish = ConvLayer(in_channels=32, out_channels=1, kernel_size=3, stride=1)

    def forward(self, input, z):
        # apply linear style

        # 4 x 4
        out = F.upsample(input, scale_factor=1/8, mode='bilinear')
        out = self.initial(out)
        out = self.upsample_1(out, z)

        # 8 x 8
        image = F.upsample(input, scale_factor=1/4, mode='bilinear')
        image = self.origin_2(image)
        out = out + image
        out = self.upsample_2(out,z)

        # 16 x 16
        image = F.upsample(input, scale_factor=1/2, mode='bilinear')
        image = self.origin_3(image)
        out = out + image
        out = self.upsample_3(out,z)

        # 32 x 32
        image = self.origin_4(input)
        out = out + image
        out = self.upsample_4(out,z)

        # 64 x 64
        image = self.origin_5(input)
        out = out + image
        out = self.upsample_5(out,z)

        # 128 x 128
        image = self.origin_6(input)
        out = out + image
        out = self.upsample_6(out,z)

        # 256 x 256
        out = self.upsample_7(out)

        # 512 x 512
        out = self.finish(out)


        return out
