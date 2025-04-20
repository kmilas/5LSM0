import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Optional, Tuple

image_size = (672,672)

class Conv(nn.Module):
    def __init__(self,
                in_feat,
                out_feat,
                kernel,
                padding=1,
                act=nn.GELU,
                bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_feat, out_feat, kernel, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(out_feat)
        self.act =  act()


    def forward(self, x):
      x = self.conv(x)
      x = self.norm(x)
      x = self.act(x)
      return x

# based on mmseg implementation https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/necks/featurepyramid.py
class Feature2Pyramid(nn.Module):
    def __init__(self,
                 embed_dim=384,
                 hidden_size = 384,
                 rescales=[4, 2, 1, 0.5]):
        super().__init__()
        self.rescales = rescales

        for k in self.rescales:
            if k == 4:
                self.up4 = nn.Sequential(*[
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(embed_dim, hidden_size, 3, 1, 1, bias=False),
                    nn.GELU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(hidden_size, hidden_size, 3, 1, 1, bias=False)
                ])
            elif k == 2:
                self.up2 = nn.Sequential(*[
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(embed_dim, hidden_size, 3, 1, 1, bias=False)
                ])
            elif k == 1:
                self.same = nn.Conv2d(embed_dim, hidden_size, 3, 1, 1, bias=False)
            elif k == 0.5:
                self.down2 = nn.Sequential(*[nn.AvgPool2d(kernel_size=2, stride=2),
                                             nn.Conv2d(embed_dim, hidden_size, 3, 1, 1, bias=False)
                ])

            else:
                raise KeyError(f'invalid {k} for feature2pyramid')

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []
        ops = [self.up4, self.up2, self.same, self.down2]

        for i in range(len(inputs)):
            outputs.append(ops[i](inputs[i]))
        return tuple(outputs)

class PPMBlock(nn.Module):
    def __init__(self, pool_scale, in_feat, out_feat):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_scale)
        self.conv_block = Conv(in_feat, out_feat, kernel=1, padding=0)


    def forward(self, input):
        x = self.pool(input)
        x = self.conv_block(x)
        return x

class PPM(nn.Module):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.
    https://arxiv.org/abs/1612.01105
    """

    def __init__(self, pool_scales, in_channels, channels):
        super().__init__()
        self.pool_scales = pool_scales

        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        for pool_scale in pool_scales:
            self.blocks.append(PPMBlock(pool_scale=pool_scale, in_feat=in_channels, out_feat=channels))

        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ppm_outs = [x]
        for ppm in self.blocks:
            ppm_out = ppm(x)
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=False
            )
            ppm_outs.append(upsampled_ppm_out)
        return torch.cat(ppm_outs, dim=1)

#in_tensor = torch.randn(2,768,16,16)
#model = PPM(pool_scales=(1,2,3,6), in_channels=768, channels=256)
#model.eval()
#out = model(in_tensor)
#print(out.shape)

class UPerHead(nn.Module):

  """
  Head: Unified Perceptual Parsing for Scene Understanding. https://arxiv.org/abs/1807.10221
  """
  # [768,768,768,768] [384, 384, 384, 384]
  def __init__(self, n_cls=19, in_channels=[384, 384, 384, 384], channels=384, pool_scales =(1, 2, 3, 6) ):
        super().__init__()
        #self.config = config
        self.pool_scales = pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = False

        # Pyramid Pooling Module
        self.ppm = PPM(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
        )

        self.bottleneck = Conv(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = Conv(in_channels, self.channels, kernel=1)
            fpn_conv = Conv(self.channels, self.channels, kernel=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = Conv(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel=3,
            padding=1,
        )
  def forward(self, encoder_hidden_states):  
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        laterals.append(self.bottleneck(self.ppm(encoder_hidden_states[-1])))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=False
            )

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=True
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)


        return output  

class FCNHead(nn.Module):
  """
  Head of Fully Convolution Networks for Semantic Segmentation. https://arxiv.org/abs/1411.4038
  """
  def __init__(
        self, n_cls=19, channels=384, in_index: int = 2, kernel_size: int = 3):
        super().__init__()

        self.in_channels = channels
        self.channels = channels
        self.num_convs = 2
        self.concat_input = True

        conv_padding = (kernel_size // 2) 
        convs = []
        convs.append(
            Conv(self.in_channels, self.channels, kernel=kernel_size, padding=conv_padding)
        )
        for i in range(self.num_convs - 1):
            convs.append(
                Conv(self.channels, self.channels, kernel=kernel_size, padding=conv_padding)
            )
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = Conv(
                self.in_channels + self.channels, self.channels, kernel=kernel_size, padding=kernel_size // 2
            )

        self.classifier = nn.Conv2d(self.channels, n_cls, kernel_size=1)

  def forward(self, hidden_states):
        # last feature map -> hidden_states
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))
        output = self.classifier(output)
        return output


class Dinov2Uper(nn.Module):
  def __init__(self, n_cls, patch_size, d_encoder, freeze=False, dropout=0.1, backbone='dinov2_vits14_reg', depth_anything=False):
        super().__init__()

        self.freeze = freeze
        self.vit = torch.hub.load('facebookresearch/dinov2', backbone)

        if depth_anything:
            d = torch.load('depth/depth_anything_v2_vitb.pth', map_location='cpu', weights_only=True)
            new_d = {}
            for key, value in d.items():
                if 'pretrained' in key:
                    new_d[key.replace('pretrained.', '')] = value
            self.vit.load_state_dict(new_d)

        self.neck = Feature2Pyramid()
        # fpn_bottleneck len(in_channels)*channels
        if self.training:
          self.aux_head = FCNHead()

        self.decode = UPerHead()
        # dropout forgot it
        self.dropout = nn.Dropout2d(dropout)
        self.cls_seg = nn.Conv2d(d_encoder, n_cls, kernel_size=1)

  def forward(self,x):
        if self.freeze:
            with torch.no_grad():
                x = self.vit.get_intermediate_layers(x,
                                        n= [3,5,7,11],  # Layers or n last layers to take
                                        reshape= True,
                                        return_class_token = False)
        else:
            x = self.vit.get_intermediate_layers(x,
                                        n= [3,5,7,11],  # Layers or n last layers to take
                                        reshape= True,
                                        return_class_token = False)

        feature_pyramid = self.neck(x)
        out = self.decode(feature_pyramid)
        # dropout
        out = self.dropout(out)
        out = self.cls_seg(out)
        if self.training:
          aux_logits = self.aux_head(feature_pyramid[-1])
          aux_logits = F.interpolate(aux_logits, size=image_size, mode="bilinear")
        out = F.interpolate(out, size=image_size, mode="bilinear")

        if self.training:
          return out, aux_logits
        else: return out

#in_tensor = torch.randn(2,3,560,560)
#model = Dinov2Uper(n_cls=19, patch_size=14, d_encoder=384)
#model.train()
#out, aux_logits = model(in_tensor)
#print(out.shape)
#print(aux_logits.shape)

