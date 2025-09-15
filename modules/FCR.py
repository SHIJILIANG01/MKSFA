import torch
import torch.nn as nn
import random

_reduction_modes = ['none', 'mean', 'sum']

class CoherenceLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(CoherenceLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.reduction = reduction
    def compute_coherence(self, im1, im2):
        pxy = (torch.fft.rfft2(im1) * torch.fft.rfft2(im2).conj() ).abs().mean((1,2,3))
        pxx = (torch.fft.rfft2(im1) * torch.fft.rfft2(im1).conj() ).abs().mean((1,2,3))
        pyy = (torch.fft.rfft2(im2) * torch.fft.rfft2(im2).conj() ).abs().mean((1,2,3))
        return 1 - (pxy) / (pxx**0.5 * pyy**0.5)
    def forward(self, pred, target, **kwargs):
        return self.compute_coherence(
            pred, target).mean()

def sample_with_j(k, n, j):
    if n >= k:
        raise ValueError("n must be less than k.")
    if j < 0 or j > k:
        raise ValueError("j must be in the range 0 to k.")

    # 创建包含0到k的数字的列表
    numbers = list(range(k))

    # 确保j在数字列表中
    if j not in numbers:
        raise ValueError("j must be in the range 0 to k.")

    # 从数字列表中选择j
    sample = [j]

    # 从剩余的数字中选择n-1个
    remaining = [num for num in numbers if num != j]
    sample.extend(random.sample(remaining, n - 1))

    return sample


class FCR(nn.Module):
    def __init__(self, ablation=False):

        super(FCR, self).__init__()
        self.l1 = nn.L1Loss()
        self.multi_n_num = 2
        self.ab = ablation

    def forward(self, a, p, n):
        a_fft = torch.fft.fft2(a)
        p_fft = torch.fft.fft2(p)
        n_fft = torch.fft.fft2(n)

        contrastive = 0
        for i in range(a_fft.shape[0]):
            d_ap = self.l1(a_fft[i], p_fft[i])
            if not self.ab:
                for j in sample_with_j(a_fft.shape[0], self.multi_n_num, i):
                    d_an = self.l1(a_fft[i], n_fft[j])
                    contrastive += (d_ap / (d_an + 1e-7))
                contrastive = contrastive / (self.multi_n_num * a_fft.shape[0])
            else:
                contrastive = d_ap

        return contrastive

if __name__ == '__main__':
    print(sample_with_j(10, 5, 5))