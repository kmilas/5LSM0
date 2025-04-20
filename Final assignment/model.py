import torch
import timm
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, padding=0, bias=False):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(in_feat, in_feat, kernel_size=kernel_size, padding=padding, groups=in_feat, bias=bias)
        self.pointwise = nn.Conv2d(in_feat, out_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

def dsconv_block_3x3(in_feat, out_feat, activ=nn.ReLU ,bias=False):
  return nn.Sequential(
      DSConv(in_feat, out_feat, 3,1,1, bias=bias),
      nn.BatchNorm2d(out_feat),
      activ()
  )

def conv_block_3x3(in_feat, out_feat, activ=nn.ReLU ,bias=False):
  return nn.Sequential(
      nn.Conv2d(in_feat, out_feat, 3,1,1, bias=bias),
      nn.BatchNorm2d(out_feat),
      activ()
  )


def conv_block_1x1(in_feat, out_feat, activ=nn.ReLU ,bias=False):
  return nn.Sequential(
      nn.Conv2d(in_feat, out_feat, 1, bias=bias),
      nn.BatchNorm2d(out_feat),
      activ()
  )

class CA(nn.Module):
    '''
    Channel Attention Layer: https://arxiv.org/pdf/1709.01507
    '''
    def __init__(self, in_feat,redu):
        super(CA,self).__init__()

        out_feat = in_feat//redu
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_feat,out_feat,1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_feat,in_feat,1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avgpool(x)
        y = self.conv(y)

        return y*x

class PPM(nn.Module):

    def __init__(self, in_feat, redu_feat, bins):
        super(PPM,self).__init__()

        self.ppm = []

        for bin in bins:
            self.ppm.append(nn.Sequential(
                  nn.AdaptiveAvgPool2d(bin),
                  nn.Conv2d(in_feat, redu_feat, kernel_size=1, bias=False),
                  nn.BatchNorm2d(redu_feat),
                  nn.ReLU(inplace=True)))
                #conv_block_1x1(in_feat, redu_feat))


        self.ppm = nn.ModuleList(self.ppm)

    def forward(self, x):
        x_size = x.size() # hxw
        out = [x]

        # Execute ever pooling layer in the pyramid and upsample the output to the input size
        i=0
        for pp in self.ppm:
            out.append(F.interpolate(pp(x), size=x_size[2:], mode='bilinear', align_corners=True))

        return torch.cat(out, dim=1)

class EfficientPPM(nn.Module):
    def __init__(self, in_feat, bins):
      super(EfficientPPM,self).__init__()

      self.ppm = []
      self.conv1x1 = conv_block_1x1(in_feat, in_feat//2)
      ppm_features = in_feat//2 * len(bins)
      self.conv1x1_ppm = conv_block_1x1(ppm_features, in_feat//2)
      self.ca = CA(in_feat//2,16)
      for bin in bins:
          self.ppm.append(nn.Sequential(
                  nn.AdaptiveAvgPool2d(bin),
                  nn.Conv2d(in_feat//2, in_feat//2, kernel_size=1, bias=False),
                  nn.BatchNorm2d(in_feat//2),
                  nn.ReLU(inplace=True)))

      self.ppm = nn.ModuleList(self.ppm)

    def forward(self, x):
      x = self.conv1x1(x)
      x = self.ca(x)
      x_size = x.size() # hxw
      print(x.shape)
      out = []
      for pp in self.ppm:
          #out.append(self.ca(pp(x)))
          ppm = self.ca(pp(x))
          out.append(F.interpolate(ppm, size=x.size()[2:], mode='bilinear', align_corners=True))
          #
      out = torch.cat(out, dim=1)
      out = self.conv1x1_ppm(out)
      #out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)
      y = torch.cat((out,x), dim=1)

      return y

class PSPNet(nn.Module):
    '''
    Implementation of PSPNet https://arxiv.org/pdf/1612.01105
    
    timm1: seresnextaa101d_32x8d.sw_in12k_ft_in1k_288
    timm2:
    '''
    def __init__(self, bins=(1,2,3,6), classes=19, backbone='seresnextaa101d_32x8d.sw_in12k_ft_in1k_288' ,pretrained=True):
        super(PSPNet,self).__init__()
        # backbone :ResNet strikes back: An improved training procedure in timm: https://arxiv.org/abs/2110.00476
        # seresnextaa101d_32x8d.sw_in12k_ft_in1k_288 last map 2048x9x9
        # resnet50d.ra4_e3600_r224_in1k last map 2048x7x7
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0) # features_only=True

        in_feat = 2048
        out_feat = int(in_feat/len(bins))
        self.ppm = PPM(in_feat,out_feat, bins)

        in_feat *=2 #if PPM
        self.decoder = nn.Sequential(
            conv_block_3x3(in_feat,512),
            nn.Conv2d(512,classes, 1))

    def forward(self, x):
        x_size = x.size()[2:] # hxw
        x = self.backbone.forward_features(x)
        x = self.ppm(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=x_size, mode='bilinear', align_corners=True)

        return x

class EPSPNet(nn.Module):
    def __init__(self, bins=(2,3,6), classes=19):
        super(EPSPNet,self).__init__()
        # training recipe :ResNet strikes back: An improved training procedure in timm: https://arxiv.org/abs/2110.00476
        # hyperparameter tuning inspired by MobileNetV4 -- Universal Models for the Mobile Ecosystem: https://arxiv.org/abs/2404.10518
        self.backbone = timm.create_model('mobilenetv4_hybrid_medium.ix_e550_r256_in1k', pretrained=True, num_classes=0)
        #self.backbone = timm.create_model('mobilenetv4_hybrid_large.e600_r384_in1k', pretrained=True, num_classes=0)
        in_feat = 960
        #in_feat = 2048

        self.ppm = EfficientPPM(in_feat, bins)

        self.decoder = nn.Sequential(
            dsconv_block_3x3(in_feat,512),
            DSConv(512,classes, 1))

    def forward(self, x):
        x_size = x.size()[2:] # hxw
        x = self.backbone.forward_features(x)
        x = self.ppm(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=x_size, mode='bilinear', align_corners=True)

        return x
