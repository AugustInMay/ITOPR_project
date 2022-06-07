################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# CNN setup and data normalization
#
################

import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, str_ = 2, pad=1, dropout=0., sf=2):
    block = nn.Sequential()

    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=False))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=False))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=str_, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=sf, mode='bilinear')) # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))
    return block

    
# generator model
class UNet_(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(UNet_, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(4, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True, relu=False, dropout=dropout )
        self.layer2b= blockUNet(channels*2, channels*4, 'layer2b',transposed=False, bn=True, relu=False, dropout=dropout, size=3, str_=5)
        self.layer3 = blockUNet(channels*4, channels*8, 'layer3', transposed=False, bn=True, relu=False, dropout=dropout, size=5, str_=5)
        # note the following layer also had a kernel size of 2 in the original version (cf https://arxiv.org/abs/1810.08217)
        # it is now changed to size 4 for encoder/decoder symmetry; to reproduce the old/original results, please change it to 2
     
        # note, kernel size is internally reduced by one now
        self.dlayer3 = blockUNet(channels*8, channels*4, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout, sf = 5)
        self.dlayer2b= blockUNet(channels*8, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout, sf = 5)
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, 4, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)

        out2 = self.layer2(out1)
        out2b= self.layer2b(out2)
        out3 = self.layer3(out2b)

        dout3 = self.dlayer3(out3)

        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)

        return dout1


class DefNet_(nn.Module):
    def __init__(self):
        super(DefNet_, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(100, 150),
            nn.ReLU(),
            nn.Linear(150, 200),
            nn.ReLU(),
            nn.Linear(200, 250),
            nn.ReLU(),
            nn.Linear(250, 200),
            nn.ReLU(),
            nn.Linear(200, 150),
            nn.ReLU(),
            nn.Linear(150, 100)
        )

    def forward(self, x):
        return self.layers(x)

class EncDec_(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(EncDec_, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(4, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True, relu=False, dropout=dropout )
        self.layer2b= blockUNet(channels*2, channels*4, 'layer2b',transposed=False, bn=True, relu=False, dropout=dropout, size=3, str_=5)
        self.layer3 = blockUNet(channels*4, channels*8, 'layer3', transposed=False, bn=True, relu=False, dropout=dropout, size=5, str_=5)
        # note the following layer also had a kernel size of 2 in the original version (cf https://arxiv.org/abs/1810.08217)
        # it is now changed to size 4 for encoder/decoder symmetry; to reproduce the old/original results, please change it to 2
     
        # note, kernel size is internally reduced by one now
        self.dlayer3 = blockUNet(channels*8, channels*4, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout, sf = 5)
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout, sf = 5)
        self.dlayer2 = blockUNet(channels*2, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels, 4, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)

        out2 = self.layer2(out1)
        out2b= self.layer2b(out2)
        out3 = self.layer3(out2b)

        dout3 = self.dlayer3(out3)

        dout2b = self.dlayer2b(dout3)
        dout2 = self.dlayer2(dout2b)
        dout1 = self.dlayer1(dout2)

        return dout1