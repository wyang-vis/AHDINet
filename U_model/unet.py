
from torch import nn
import scipy.stats as st
import torch
from .net_util import *
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((1, 1, kernlen, kernlen))
    out_filter = np.repeat(out_filter, channels, axis = 0)
    return out_filter





class EEC(nn.Module):
    def __init__(self, dim, bias=False):
        super(EEC, self).__init__()


        self.Conv=nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)
        self.CA=ChannelAttention(dim)


    def forward(self, f_img, f_event,Mask):


        assert f_img.shape == f_event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = f_img.shape

        F_event=f_event*Mask
        F_event=f_event+F_event
        F_cat = torch.cat([F_event, f_img], dim=1)
        F_conv=self.Conv(F_cat)
        w_c=self.CA(F_conv)
        F_event=F_event*w_c
        F_out=F_event+f_img

        return F_out



class ISC(nn.Module):
    def __init__(self, dim, num_heads=4,  bias=False, LayerNorm_type='WithBias'):
        super(ISC, self).__init__()
        self.num_heads = num_heads
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.SA= Spatio_Attention(dim, num_heads, bias)
        self.CA=ChannelAttention(dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(2*dim, dim // 16, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(dim // 16, dim, 1, bias=False)
        self.sigmoid= nn.Sigmoid()
    def forward(self, f_img, f_event):

        assert f_img.shape == f_event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = f_img.shape
        SA_att,V=self.SA(f_img)
        F_img=(V@SA_att)
        F_img = rearrange(F_img, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        CA_att=self.CA(f_img)
        F_img=F_img*CA_att
        F_img=F_img+f_img
        w_i=self.avg_pool(F_img)
        w_e=self.avg_pool(f_event)
        w=torch.cat([w_i,w_e],dim=1)
        w= self.fc2(self.relu1(self.fc1(w)))
        w=self.sigmoid(w)
        F_img=F_img*w
        F_event=f_event*(1-w)
        F_event=F_event+F_img

        return F_event




class Decoder(nn.Module):
    """Modified version of Unet from SuperSloMo.
    """

    def __init__(self, channels):
        super(Decoder, self).__init__()
        ######Decoder
        self.up1 = DE_Block(channels[3], channels[2])
        self.up2 = DE_Block(channels[2], channels[1])
        self.up3 = DE_Block(channels[1], channels[0])

    def forward(self, input):
        x4=input[3]
        x3 = self.up1(x4, input[2])
        x2 = self.up2(x3, input[1])
        x1 = self.up3(x2, input[0])
        return x1





class Restoration(nn.Module):
    """Modified version of Unet from SuperSloMo.

    """

    def __init__(self, inChannels_img, inChannels_event,outChannels, args,ends_with_relu=False):
        super(Restoration, self).__init__()
        self._ends_with_relu = ends_with_relu
        self.num_heads=4
        self.act = nn.ReLU(inplace=True)

        self.channels = [64,64, 128, 256]

        self.encoder_img_1=EN_Block(inChannels_img, self.channels[0],0)
        self.encoder_img_2=EN_Block(self.channels[0], self.channels[1],0)
        self.encoder_img_3=EN_Block(self.channels[1], self.channels[2],0)
        self.encoder_img_4=EN_Block(self.channels[2], self.channels[3],1)


        self.encoder_event_1=EN_Block(inChannels_event, self.channels[0],0)
        self.encoder_event_2=EN_Block(self.channels[0], self.channels[1],0)
        self.encoder_event_3=EN_Block(self.channels[1], self.channels[2],0)
        self.encoder_event_4=EN_Block(self.channels[2], self.channels[3],1)


        self.down = DownSample()



        self.EEC_1=EEC(self.channels[0])
        self.EEC_2=EEC(self.channels[1])

        self.ISC_3=ISC(self.channels[2])
        self.ISC_4=ISC(self.channels[3])


        self.decoder_img = Decoder(self.channels)
        self.decoder_event = Decoder(self.channels)
        self.weight_fusion=Weight_Fusion(self.channels[0])


        self.out = nn.Conv2d(self.channels[0], outChannels, 3, stride=1, padding=1)

    def blur(self, x, kernel = 21, channels = 3, stride = 1, padding = 'same'):
        kernel_var = torch.from_numpy(gauss_kernel(kernel, 3, channels)).to(device).float()
        return torch.nn.functional.conv2d(x, kernel_var, stride = stride, padding = int((kernel-1)/2), groups = channels)

    def forward(self, input_img, input_event):


        M0 = torch.clamp(self.blur(self.blur(torch.sum(torch.abs(input_event), axis = 1, keepdim = True), kernel = 7, channels = 1),
                                            kernel = 7, channels = 1), 0, 1)

        img_encoder_list = []
        event_encoder_list = []

        ####  feature extraction

        img_1=self.encoder_img_1(input_img)
        event_1=self.encoder_event_1(input_event)
        img_encoder_list.append(img_1)
        event_encoder_list.append(event_1)

        down_img_1=self.down(img_1)
        down_event_1=self.down(event_1)
        M1 = self.blur(M0, kernel=5, channels=1, padding=2, stride=2)
        fuse_img_1=self.EEC_1(down_img_1,down_event_1,M1)


        img_2=self.encoder_img_2(fuse_img_1)
        event_2=self.encoder_event_2(down_event_1)
        img_encoder_list.append(img_2)
        event_encoder_list.append(event_2)

        down_img_2=self.down(img_2)
        down_event_2=self.down(event_2)
        M2 = self.blur(M1, kernel=5, channels=1, padding=2, stride=2)
        fuse_img_2=self.EEC_2(down_img_2,down_event_2,M2)



        img_3=self.encoder_img_3(fuse_img_2)
        event_3=self.encoder_event_3(down_event_2)
        img_encoder_list.append(img_3)
        event_encoder_list.append(event_3)

        down_img_3=self.down(img_3)
        down_event_3=self.down(event_3)
        fuse_event_3= self.ISC_3(down_img_3,down_event_3)



        img_4=self.encoder_img_4(down_img_3)
        event_4=self.encoder_event_4(fuse_event_3)
        event_4= self.ISC_4(img_4,event_4)
        img_encoder_list.append(img_4)
        event_encoder_list.append(event_4)


        de_img=self.decoder_img(img_encoder_list)
        de_event=self.decoder_event(event_encoder_list)
        de_fuse=self.weight_fusion(de_img,de_event)
        out=self.out(de_fuse)

        out=out+input_img
        return out
