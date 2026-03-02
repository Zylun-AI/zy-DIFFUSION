import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True)
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.proj = None

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.use_attention:
            b, c, h, w = out.shape
            out_ = out.view(b, c, h*w).transpose(1, 2)
            out_, _ = self.attn(out_, out_, out_)
            out = out_.transpose(1, 2).view(b, c, h, w)
        return self.act1(out + identity)

class ImprovedUNet(nn.Module):
    def __init__(self, in_ch=4, ch=256, ch_mult=[1,2,4,8,8], attn_layers=[2,3,4], text_dim=768, image_size=32):
        super().__init__()
        self.inc = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        channels = [ch * m for m in ch_mult]
        # encoder
        in_channels = ch
        for i, out_channels in enumerate(channels):
            self.enc_blocks.append(
                ResidualBlock(in_channels, out_channels, use_attention=(i in attn_layers))
            )
            self.downs.append(nn.Conv2d(out_channels, out_channels, 4, 2, 1))
            in_channels = out_channels
        # middle block
        self.mid = ResidualBlock(in_channels, in_channels, use_attention=True)
        # decoder
        for i, out_channels in reversed(list(enumerate(channels))):
            self.dec_blocks.append(
                ResidualBlock(in_channels*2, out_channels, use_attention=(i in attn_layers))
            )
            self.ups.append(nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1))
            in_channels = out_channels
        self.outc = nn.Conv2d(ch, 4, 3, padding=1)
        # Text embedding adaptation
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, image_size*image_size),
            nn.SiLU()
        )

    def forward(self, x, text_emb):
        x1 = self.inc(x)
        enc_feats = []
        h = x1
        for enc, down in zip(self.enc_blocks, self.downs):
            h = enc(h)
            enc_feats.append(h)
            h = down(h)
        h = self.mid(h)
        for dec, up, enc_feat in zip(self.dec_blocks, self.ups, reversed(enc_feats)):
            h = torch.cat([h, enc_feat], dim=1)
            h = dec(h)
            h = up(h)
        # Add text embedding as conditioning
        b, _, h_, w_ = h.shape
        text_map = self.text_proj(text_emb).view(b, 1, h_, w_)
        h = h + text_map
        return self.outc(h)