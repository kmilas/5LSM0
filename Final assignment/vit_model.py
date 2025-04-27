import torch
from einops import rearrange
from timm.layers import trunc_normal_
import torch.nn as nn
import torch.nn.functional as F

image_size = (560,560)

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x

class Segmenter(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder, backbone='dinov2_vitb14_reg', depth_anything=False):
        super().__init__()

        self.vit = torch.hub.load('facebookresearch/dinov2', backbone)

        if depth_anything:
            d = torch.load('depth/depth_anything_v2_vitb.pth', map_location='cpu', weights_only=True)
            new_d = {}
            for key, value in d.items():
                if 'pretrained' in key:
                    new_d[key.replace('pretrained.', '')] = value
            self.vit.load_state_dict(new_d)

        self.decoder = DecoderLinear(n_cls, patch_size, d_encoder)

    def forward(self,x, im_size=(560,560)):
        x = self.vit.forward_features(x)['x_norm_patchtokens']
        x = self.decoder(x, im_size)
        x = F.interpolate(x, size=im_size, mode="bilinear")
        return x
