import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
 

def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
 
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
 
    def forward(self, x):
        x = self.conv(x)
        return x
 
class up_conv(nn.Module):
 
    def __init__(self, in_ch, out_ch, scale_factor = 2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        x = self.up(x)
        return x

class fine_decoder(nn.Module):
    def __init__(self, dim_in=768, multi_sup=False, out_size=518):
        super(fine_decoder, self).__init__()

        self.dim_in = dim_in
        self.multi_sup = multi_sup
        self.out_size = out_size

        self.org_conv = conv_block(dim_in*2, dim_in // 2)
        self.diff_conv = conv_block(dim_in*2, dim_in // 2)
        self.fusion_conv = conv_block(dim_in, dim_in)

        self.Up1 = up_conv(dim_in, dim_in // 2)
        self.Up2 = up_conv(dim_in // 2, dim_in // 4)
        self.Up3 = up_conv(dim_in // 4, dim_in // 8)
        self.seg_head_3 = nn.Conv2d(dim_in // 8, 2, kernel_size=1, stride=1, padding=0)
        if self.multi_sup:
            self.seg_head_2 = nn.Conv2d(dim_in // 4, 2, kernel_size=1, stride=1, padding=0)
            self.seg_head_1 = nn.Conv2d(dim_in // 2, 2, kernel_size=1, stride=1, padding=0)

        self.apply(init_weight)

    def forward(self, en_f_list, de_f_list):
        en_f_list = torch.cat(en_f_list, dim=1) # [16, 768*2, 37, 37]
        de_f_list = torch.cat(de_f_list, dim=1)
        diff_features = (de_f_list - en_f_list).detach()    # [16, 768*2, 37, 37]

        org_features_ = self.org_conv(en_f_list)
        diff_features_ = self.diff_conv(diff_features)
        fusion_features = self.fusion_conv(torch.cat([org_features_, diff_features_], dim=1))

        out_list = []
        f = self.Up1(fusion_features)   # 74
        if self.multi_sup:
            out_1 = self.seg_head_1(f)
            out_list.append(out_1.softmax(1))
        f = self.Up2(f) # 148
        if self.multi_sup:
            out_2 = self.seg_head_2(f)
            out_list.append(out_2.softmax(1))
        f = self.Up3(f) # 296
        f = self.seg_head_3(f)
        out = F.interpolate(f, size=self.out_size, mode='bilinear', align_corners=True)
        out_list.append(out.softmax(1))

        return out_list


from models.model_utils import ASPP, BasicBlock, l2_normalize, make_layer
class SegmentationNet(nn.Module):
    def __init__(self, inplanes=768*2):
        super().__init__()
        self.res = make_layer(BasicBlock, inplanes, 256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head = nn.Sequential(
            ASPP(256, 256, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, en_f_list, de_f_list):
        en_f_list = torch.cat(en_f_list, dim=1) # [16, 768*2, 37, 37]
        de_f_list = torch.cat(de_f_list, dim=1)
        x = (de_f_list - en_f_list).detach()
        x = self.res(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x