#losses.py 
import torch 
import torch.nn as nn 
import torch.nn.functional as F


def _gaussian_window(window_size= 11, sigma=1.5):
    coords= torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    win = g[:, None] * g[None, :]
    return win[None, None]

def ssim(pred, target, data_range=1.0, window_size=11):
    win = _gaussian_window(window_size).to(pred.device, pred.dtype)
    pad = window_size // 2 

    mu_p = F.conv2d(pred, win, padding=pad)
    mu_t = F.conv2d(target, win, padding=pad)
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

    sig_p2 = F.conv2d(pred * pred,     win, padding=pad) - mu_p2
    sig_t2 = F.conv2d(target * target, win, padding=pad) - mu_t2
    sig_pt = F.conv2d(pred * target,   win, padding=pad) - mu_pt

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    num = (2 * mu_pt + C1) * (2 * sig_pt + C2)
    den = (mu_p2 + mu_t2 + C1) * (sig_p2 + sig_t2 + C2)
    return (num / den).mean()

class L1SSIMLoss(nn.Module):
    def __init__(self, ssim_weight=0.1, data_range=1.0):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.data_range = data_range

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        s = ssim(pred, target, data_range=self.data_range)
        loss = l1 + self.ssim_weight * (1.0 - s)
        return loss, {"l1": l1.detach(), "ssim": s.detach()}