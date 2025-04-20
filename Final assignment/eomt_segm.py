import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Optional, Tuple

class DinoV2WithInjectedClassTokens(nn.Module):
    def __init__(self, num_classes, image_size=(560, 560)):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.cls_emb = nn.Parameter(torch.randn(1, num_classes, self.vit.embed_dim))

        # Split blocks for early and late processing
        self.early_blocks = self.vit.blocks[:-4]
        self.late_blocks = self.vit.blocks[-4:]

    def forward(self, x):
        B = x.shape[0]

        H, W = self.image_size

        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.vit.interpolate_pos_encoding(x, W, H)
        #x = self.vit.norm_pre(x)

        # Early blocks (no class tokens yet)
        for blk in self.early_blocks:
            x = blk(x)

        # Inject class tokens
        cls_tokens = self.cls_emb.expand(B, -1, -1)  # (B, num_classes, C)
        x = torch.cat([x, cls_tokens], dim=1)

        # Late blocks
        for blk in self.late_blocks:
            x = blk(x)

        x = self.vit.norm(x)

        return x
  
class SegmenterStyleDecoder(nn.Module):
    def __init__(self, patch_size, d_model, num_classes):
        super().__init__()
        self.patch_size = patch_size
        self.n_cls = num_classes
        self.proj_patch = nn.Parameter(torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(torch.randn(d_model, d_model))
        self.mask_norm = nn.LayerNorm(num_classes)

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        patches, cls_tokens = x[:, 1:-self.n_cls], x[:, -self.n_cls:]
        patches = patches @ self.proj_patch
        cls_tokens = cls_tokens @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_tokens = cls_tokens / cls_tokens.norm(dim=-1, keepdim=True)

        masks = patches @ cls_tokens.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=GS)

        return masks

  
class SegmenterEoMT(nn.Module):
    '''
    Inspired by Segmenter mask transformer https://arxiv.org/abs/2105.05633
    https://github.com/rstrudel/segmenter and
    EoMT https://arxiv.org/abs/2503.19108
    '''
    def __init__(self, num_classes=19, patch_size=14, d_model=768, image_size=(560, 560)):
        super().__init__()
        self.encoder = DinoV2WithInjectedClassTokens(num_classes, image_size)
        self.decoder = SegmenterStyleDecoder(patch_size, d_model, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        masks = self.decoder(x, self.encoder.image_size)
        masks = F.interpolate(masks, size=self.encoder.image_size, mode='bilinear', align_corners=False)
        return masks
  
#x = torch.randn(2, 3, 560, 560)
#model = SegmenterEomT()
#out = model(x)
#print(out.shape)