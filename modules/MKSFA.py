from modules.MSAB import MSABlock
from modules.wtconv2d import *
from modules.FDCAB import FDCABlock

class MKSFANet(nn.Module):
    def __init__(self, num_blocks=[1, 1, 1, 1]):
        super(MKSFANet, self).__init__()

        self.conv1 = nn.Sequential(
            WT_downsample(3, 32, kernel_size=7, stride=1, wt_levels=3),
            nn.BatchNorm2d(32),
            nn.SiLU())
        self.encoder1 = nn.Sequential(*[MSABlock(32, 0.5, 2.21, 2) for i in range(num_blocks[0])])

        self.conv2 = nn.Sequential(
            WT_downsample(32, 64, kernel_size=3, stride=2, wt_levels=3),
            nn.BatchNorm2d(64),
            nn.SiLU())
        self.encoder2 = nn.Sequential(*[MSABlock(64, 0.5, 2.21, 3) for i in range(num_blocks[1])])

        self.conv3 = nn.Sequential(
            WT_downsample(64, 128, kernel_size=3, stride=2, wt_levels=3),
            nn.BatchNorm2d(128),
            nn.SiLU())
        self.encoder3 = nn.Sequential(*[MSABlock(128, 0.5, 2.21, 4) for i in range(num_blocks[2])])

        self.middle_conv = nn.Sequential(
            WT_downsample(128, 256, kernel_size=3, stride=2, wt_levels=3),
            nn.BatchNorm2d(256),
            nn.SiLU())
        self.middle = nn.Sequential(*[MSABlock(256, 0.5, 2.21, 5) for i in range(num_blocks[3])])

        self.Tconv3 = nn.Sequential(
            WT_upsample(256, 128, kernel_size=3, stride=1, wt_levels=3),
            nn.BatchNorm2d(128),
            nn.SiLU())
        self.decoder3 = nn.Sequential(*[
            FDCABlock(128, 2.21) for i in range(num_blocks[2])])

        self.Tconv2 = nn.Sequential(
            WT_upsample(128, 64, kernel_size=3, stride=1, wt_levels=3),
            nn.BatchNorm2d(64),
            nn.SiLU())
        self.decoder2 = nn.Sequential(*[
            FDCABlock(64, 2.21) for i in range(num_blocks[1])])

        self.Tconv1 = nn.Sequential(
            WT_upsample(64, 32, kernel_size=3, stride=1, wt_levels=3),
            nn.BatchNorm2d(32),
            nn.SiLU())
        self.decoder1 = nn.Sequential(*[
            FDCABlock(32, 2.21) for i in range(num_blocks[0])])

        self.Tconv = WT_downsample(32, 3, kernel_size=7, stride=1, wt_levels=3)


    def forward(self, input):

        x1 = self.conv1(input)
        x11 = self.encoder1(x1)
        x_skip_1 = x11

        x2 = self.conv2(x11)
        x22 = self.encoder2(x2)
        x_skip_2 = x22

        x3 = self.conv3(x22)
        x33 = self.encoder3(x3)
        x_skip_3 = x33

        out1 = self.middle_conv(x33)
        out = self.middle(out1)

        y3 = self.Tconv3(out) + x_skip_3
        y33 = self.decoder3(y3)

        y2 = self.Tconv2(y33) + x_skip_2
        y22 = self.decoder2(y2)

        y1 = self.Tconv1(y22) + x_skip_1
        y11 = self.decoder1(y1)

        y = self.Tconv(y11)
        x = torch.tanh(y)

        return x



