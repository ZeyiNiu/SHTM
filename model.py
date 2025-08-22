import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """GroupNorm → SiLU → Conv ×2 + Dropout + Residual"""
    def __init__(self, ch, dropout=0.13):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)         # B,HW,C
        k = k.reshape(b, c, h * w)                           # B,C,HW
        v = v.reshape(b, c, h * w).permute(0, 2, 1)         # B,HW,C
        attn = torch.softmax(torch.bmm(q, k) / (c ** 0.5), dim=-1)
        out  = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)

class DownStage(nn.Module):
    """Conv → 4×Res → (Attn) → DownSample"""
    def __init__(self, in_ch, out_ch, use_attn=False):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.res     = nn.Sequential(*[ResBlock(out_ch) for _ in range(4)])
        self.attn    = SelfAttention(out_ch) if use_attn else nn.Identity()
        self.down    = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res(x)
        x = self.attn(x)
        skip = x
        return self.down(x), skip


class UpStage(nn.Module):
    def __init__(self, in_ch, out_ch, use_attn=False):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res  = nn.Sequential(*[ResBlock(out_ch * 2) for _ in range(4)])
        self.attn = SelfAttention(out_ch * 2) if use_attn else nn.Identity()
        self.conv_out = nn.Conv2d(out_ch * 2, out_ch, 1)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)        # 2·out_ch
        x = self.res(x)
        x = self.attn(x)
        x = self.conv_out(x)                   # out_ch
        return x

class ERA2HiResUNet(nn.Module):
    def __init__(self, in_ch: int = 13, out_ch: int = 5):
        super().__init__()
        base = 128
        mult = [1, 2, 2, 2, 2, 2]              # 6 层
        chs  = [base * m for m in mult]        # [128,256,256,256,256,256]

        # Encoder ↓
        self.d0 = DownStage(in_ch,   chs[0])               # 209×289 → 105×145
        self.d1 = DownStage(chs[0], chs[1])                # 105×145 → 53×73
        self.d2 = DownStage(chs[1], chs[2])                # 53×73  → 27×37
        self.d3 = DownStage(chs[2], chs[3])                # 27×37  → 14×19
        self.d4 = DownStage(chs[3], chs[4], use_attn=True) # 14×19  → 7×10  (attn)
        self.d5 = DownStage(chs[4], chs[5])                # 7×10   → 4×5

        # Bottleneck
        self.mid = nn.Sequential(
            ResBlock(chs[5]),
            SelfAttention(chs[5]),
            ResBlock(chs[5])
        )

        # Decoder ↑ 
        self.u5 = UpStage(chs[5], chs[4])
        self.u4 = UpStage(chs[4], chs[3], use_attn=True)
        self.u3 = UpStage(chs[3], chs[2])
        self.u2 = UpStage(chs[2], chs[1])
        self.u1 = UpStage(chs[1], chs[1])      # 256 → 256
        self.u0 = UpStage(chs[1], chs[0])      # 256 → 128

        self.out_conv = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        # Encoder
        x, s0 = self.d0(x)
        x, s1 = self.d1(x)
        x, s2 = self.d2(x)
        x, s3 = self.d3(x)
        x, s4 = self.d4(x)
        x, s5 = self.d5(x)

        # Bottleneck
        x = self.mid(x)

        # Decoder
        x = self.u5(x, s5)
        x = self.u4(x, s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        x = self.u0(x, s0)

        x = self.out_conv(x)
        # Upsample to exactly 521×721
        return F.interpolate(x, size=(521, 721), mode='bilinear', align_corners=False)
