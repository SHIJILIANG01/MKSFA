from modules.wtconv2d import *

def make_divisible(value, divisor, min_value=None, min_ratio=0.9):

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)

    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class SAB(nn.Module):
    def __init__(self, kernel_size=3, wt_levels=3):
        super(SAB, self).__init__()

        assert kernel_size in (3, 5, 7), 'kernel must be 3 or 7 or 11'
        assert wt_levels in (3, 4, 5), 'wt_levels be 3 or 4 or 5'

        self.conv = WT_downsample(2, 1, kernel_size, wt_levels=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        out = input * self.sigmoid(x)
        return out


class MGFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor):
        super(MGFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = WT_downsample(dim, hidden_features * 2, kernel_size=1)

        self.dwconv = WTConv2d(hidden_features*2, hidden_features*2, kernel_size=3)
        self.dwconv1 = WTConv2d(hidden_features, hidden_features, kernel_size=3)
        self.dwconv2 = WTConv2d(hidden_features, hidden_features, kernel_size=5)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = x1 + self.dwconv1(x1)
        x2 = x2 + self.dwconv2(x2)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class MSAM(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=1.0, levels=3):
        super(MSAM, self).__init__()

        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.project_in = WT_downsample(in_channels, hidden_channels, kernel_size=3)
        self.wtconv1 = nn.Sequential(
            WTConv2d(hidden_channels, hidden_channels, kernel_size=5, wt_levels=levels),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU())
        self.wtconv2 = nn.Sequential(
            WTConv2d(hidden_channels, hidden_channels, kernel_size=7, wt_levels=levels),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU())
        self.wtconv3 = nn.Sequential(
            WTConv2d(hidden_channels, hidden_channels, kernel_size=9, wt_levels=levels),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU())
        self.wtconv4 = nn.Sequential(
            WTConv2d(hidden_channels, hidden_channels, kernel_size=11, wt_levels=levels),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU())
        self.project_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        self.sa = SAB(kernel_size=7, wt_levels=5)

    def forward(self, x):
        x = self.project_in(x)
        x = x + self.sa(self.wtconv1(x)) + self.sa(self.wtconv2(x)) + self.sa(self.wtconv3(x)) + self.sa(self.wtconv4(x))
        out = self.project_out(x)

        return out

class MSABlock(nn.Module):
    def __init__(self, dim, expansion=0.5, ffn_expansion_factor=2.21, levels=3):
        super(MSABlock, self).__init__()

        self.layer_norm = nn.LayerNorm(dim)
        self.Bottleneck = MSAM(dim, dim, expansion, levels)
        self.FFN = MGFN(dim, ffn_expansion_factor)

    def forward(self, input):
        b, c, h, w = input.shape
        x = self.layer_norm((input).reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)
        x_attn = input + self.Bottleneck(x)

        x = self.layer_norm((x_attn).reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)
        out = x_attn + self.FFN(x)

        return out
