from . import utils
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, stride=2):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_chans),
            nn.LeakyReLU(),

            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_chans),
            nn.LeakyReLU(),
        )

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(tensor)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})'


@utils.register_model(name='unet')
class Unet(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, chans=64, num_pool_layers=4, use_residual=True):
        super().__init__()
        # self.config = config
        # self.in_chans = config.model.in_chans
        # self.out_chans = config.model.out_chans
        # self.chans = config.model.chans
        # self.num_pool_layers = config.model.num_pool_layers
        # self.use_residual = config.model.use_residual

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = self.chans
        self.down_sample_layers = nn.ModuleList([ConvBlock(self.in_chans, self.chans, stride=1)])
        for i in range(self.num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlock(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(self.num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, self.out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)
        if self.use_residual:
            output = output + tensor

        return output