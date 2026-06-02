#u net

import torch 
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True)
    )

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=91, scalar_dim=122, out_channels = 1, base = 32, scalar_embed=64):
        super().__init__()

        #encode 
        self.enc1 = conv_block(in_channels, base)        # 32
        self.enc2 = conv_block(base,        base * 2)    # 64
        self.enc3 = conv_block(base * 2,    base * 4)    # 128
        self.enc4 = conv_block(base * 4,    base * 8)    # 256
        self.pool = nn.MaxPool2d(2)

        #scalar 
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, scalar_embed),
            nn.ReLU(inplace=True),
            nn.Linear(scalar_embed, scalar_embed),
            nn.ReLU(inplace=True),
        )
        #bottleneck 
        self.bottleneck = conv_block(base * 8 + scalar_embed, base * 16)

        #decode 
        self.up4  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = conv_block(base * 16 + base * 8, base * 8)
        self.up3  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = conv_block(base * 8  + base * 4, base * 4)
        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = conv_block(base * 4  + base * 2, base * 2)
        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = conv_block(base * 2  + base,     base)

        self.out_conv = nn.Conv2d(base, out_channels, 1)
    def forward(self, spatial, scalars):
        # encode
        e1 = self.enc1(spatial)              # (B,  32, H,    W)
        e2 = self.enc2(self.pool(e1))        # (B,  64, H/2,  W/2)
        e3 = self.enc3(self.pool(e2))        # (B, 128, H/4,  W/4)
        e4 = self.enc4(self.pool(e3))        # (B, 256, H/8,  W/8)
        p  = self.pool(e4)                   # (B, 256, H/16, W/16)

        # scalar branch -> broadcast as constant channels at bottleneck
        s = self.scalar_mlp(scalars)                              # (B, 64)
        s = s[:, :, None, None].expand(-1, -1, p.shape[2], p.shape[3])
        b = self.bottleneck(torch.cat([p, s], dim=1))             # (B, 512, H/16, W/16)

        # decode
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)        