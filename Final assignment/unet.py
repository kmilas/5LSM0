import torch
import torch.nn as nn
import timm 

class conv_block_1x1(nn.Module):
    def __init(self, in_feat, out_feat, activ=nn.ReLU ,bias=False):
        super(conv_block_1x1,self).__init()

        self.conv = nn.Conv2d(in_feat, out_feat, 1, bias=bias)
        self.batchorm = nn.BatchNorm2d(out_feat)
        self.act = activ()

    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)

        return self.act(x)

class conv_block_3x3(nn.Module):
    def __init(self, in_feat, out_feat, activ=nn.ReLU ,bias=False):
        super(conv_block_3x3,self).__init()

        self.conv = nn.Conv2d(in_feat, out_feat, 3,1,1, bias=bias)
        self.batchorm = nn.BatchNorm2d(out_feat)
        self.act = activ()

    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)

        return self.act(x)

class CA(nn.Module):
    '''
    Channel Attention Layer: https://arxiv.org/pdf/1709.01507
    '''
    def __init__(self, in_feat,redu):
        super(CA,self).__init__()

        out_feat = in_feat//redu
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_feat,out_feat, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_feat,in_feat, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.avgpool(x)
        y = self.conv(y)

        return y*x

class PPM(nn.Module):
    '''
    Pyramid Pooling Module
    '''
    def __init__(self, in_feat, redu_feat, bins):
        super(PPM,self).__init__()

        self.ppm = []

        for bin in bins:
            self.ppm += nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                conv_block_1x1(in_feat, out_feat))
        
        self.ppm = nn.ModuleList(self.ppm)
    
    def forward(self, x):
        x_size = x.size()[2:] # hxw
        out = [x]

        # Execute ever pooling layer in the pyramid and upsample the output to the input size
        for pp in self.ppm:
            pool = pp(x)
            out += F.interpolate(pool, size=x_size, mode='bilinear', align_corners=True)

        return torch.cat(out, dim=1)

class PSPNet(nn.Module):
    '''
    Implementation of PSPNet https://arxiv.org/pdf/1612.01105
    '''
    def __init__(self, bins=(1,2,3,6), classes=19):
        super(PSPNet,self).__init__()
        # training recipe :ResNet strikes back: An improved training procedure in timm: https://arxiv.org/abs/2110.00476
        # hyperparameter tuning inspired by MobileNetV4 -- Universal Models for the Mobile Ecosystem: https://arxiv.org/abs/2404.10518
        self.backbone = timm.create_model('resnet50d.ra4_e3600_r224_in1k', pretrained=True, num_classes=0) # features_only=True
        in_feat = 2048
        out_feat = int(in_feat/len(bins))
        self.ppm = PPM(in_feat,out_feat, bins)

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

class UNet(nn.Module):
    """ 
    A simple U-Net architecture for image segmentation.
    Based on the U-Net architecture from the original paper:
    Olaf Ronneberger et al. (2015), "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf
    """
    def __init__(self, in_channels=3, n_classes=1):
        
        super(UNet, self).__init__()

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 512))
        self.up1 = (Up(1024, 256))
        self.up2 = (Up(512, 128))
        self.up3 = (Up(256, 64))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits
        

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)