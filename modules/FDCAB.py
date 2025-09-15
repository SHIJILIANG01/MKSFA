from modules.wtconv2d import *
from modules.MCA import MomentAttention_v1

class MGFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor):
        super(MGFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = WT_downsample(dim, hidden_features * 2, kernel_size=3)

        self.dwconv1 = WTConv2d(hidden_features, hidden_features, kernel_size=3)
        self.dwconv2 = WTConv2d(hidden_features, hidden_features, kernel_size=5)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)

    def forward(self, x):
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        x1 = x1 + self.dwconv1(x1)
        x2 = x2 + self.dwconv2(x2)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class CB(nn.Module):
    def __init__(self, channels):
        super(CB, self).__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = nn.Sequential(
            WTConv2d(channels, channels, kernel_size=3),
            nn.SiLU())
        self.att = MomentAttention_v1()
        self.conv2 = WTConv2d(channels, channels, kernel_size=3)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(self.avg_pool(x))
        x2 = self.att(x1)
        attn = self.act(self.conv2(x2 + x))
        attn = x * attn

        return attn

class FDB(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        super(FDB, self).__init__()
        self.groups = groups
        self.conv_layer1 = WTConv2d(in_channels=in_channels * 2, out_channels=out_channels * 2, kernel_size=3)
        self.conv_layer2 = WTConv2d(in_channels=in_channels * 2, out_channels=out_channels * 2, kernel_size=5)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted1 = self.conv_layer1(ffted)
        ffted2 = self.conv_layer2(ffted)
        ffted = self.relu(self.bn(ffted1 + ffted2))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output

class FDCAM(nn.Module):
    def __init__(self, dim):
        super(FDCAM, self).__init__()

        self.fdb = FDB(dim, dim)
        self.cb = CB(dim)
        self.project_out = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.SiLU())

    def forward(self, x):
        x0 = x
        x = self.cb(x0) + self.fdb(x0) + x0
        x = self.project_out(x)

        return x

class FDCABlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.21):
        super(FDCABlock, self).__init__()

        self.layer_norm = nn.LayerNorm(dim)
        self.Bottleneck = FDCAM(dim)
        self.FFN = MGFN(dim, ffn_expansion_factor)

    def forward(self, input):
        b, c, h, w = input.shape
        x = self.layer_norm((input).reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)
        x_attn = input + self.Bottleneck(x)

        x = self.layer_norm((x_attn).reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)
        out = x_attn + self.FFN(x)

        return out
