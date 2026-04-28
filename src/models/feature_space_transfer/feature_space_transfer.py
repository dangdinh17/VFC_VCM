import torch
import torch.nn as nn
from src.models.layers import *

class BottleneckBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, activate="relu"):
        super(BottleneckBlock, self).__init__()
        
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True) if activate=="relu" else nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(mid_ch)
        self.relu2 = nn.ReLU(inplace=True) if activate=="relu" else nn.Sigmoid()
        
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm3 = nn.BatchNorm2d(out_ch)
        self.relu3 = nn.ReLU(inplace=True) if activate=="relu" else nn.Sigmoid()
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu3(out+residual)
        
        return out

class DownNet(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(DownNet, self).__init__()
        self.down = nn.Sequential(
            Down2xConv(in_channel, mid_channel),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            VSSBlock(hidden_dim=mid_channel, drop_path=0.1),
            Down2xConv(mid_channel, out_channel),   
            VSSBlock(hidden_dim=out_channel, drop_path=0.1),
        )
        
    
    def forward(self, x):
        x = self.down(x)
        return x

class UpNet(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, activate="relu"):
        super(UpNet, self).__init__()
        
        self.up = nn.Sequential(
            Up2xConv(in_channel, mid_channel),
            VSSBlock(hidden_dim=mid_channel, drop_path=0.1),
            VSSBlock(hidden_dim=mid_channel, drop_path=0.1),
            Up2xConv(mid_channel, mid_channel),
            VSSBlock(hidden_dim=mid_channel, drop_path=0.1),
            VSSBlock(hidden_dim=mid_channel, drop_path=0.1),
            Conv(mid_channel, out_channel)
        )
        
       
    def forward(self, x):
        x = self.up(x)
        return x

class FeatureSpaceTransfer(torch.nn.Module):
    def __init__(self, in_channel=256,  mid_channel=64):
        super(FeatureSpaceTransfer, self).__init__()
        
        self.branch1_up = UpNet(in_channel=in_channel, out_channel=3, mid_channel=mid_channel, activate="sigmoid")
        self.branch1_down = DownNet(in_channel=3, out_channel=in_channel, mid_channel=mid_channel)
        
        self.branch2_down = DownNet(in_channel=in_channel, out_channel=mid_channel, mid_channel=mid_channel)
        self.branch2_up = UpNet(in_channel=mid_channel, out_channel=in_channel, mid_channel=mid_channel, activate="relu")
        
        self.branch3 = nn.Sequential(
            VSSBlock(hidden_dim=in_channel, drop_path=0.1),
            VSSBlock(hidden_dim=in_channel, drop_path=0.1),
            VSSBlock(hidden_dim=in_channel, drop_path=0.1),
        )
        
    
        self.fuse = nn.Conv2d(in_channel*3, in_channel, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        recon_image = self.branch1_up(x)
        y1 = self.branch1_down(recon_image)
        
        y2 = self.branch2_down(x)
        y2 = self.branch2_up(y2)
        
        y3 = self.branch3(x)
        
        F_cur_hat = torch.cat((y1, y2, y3), dim=1)
        F_cur_hat = self.fuse(F_cur_hat)
        return F_cur_hat, recon_image
        
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# test 
if __name__ == "__main__":
    # create the model
    model = FeatureSpaceTransfer().cuda()
    print(f"Total trainable parameters in the model: {count_parameters(model)}")
    
    print(f"Total trainable parameters in the model.up: {count_parameters(model.branch1_up)}")
    print(f"Total trainable parameters in the model.down: {count_parameters(model.branch1_down)}")
    
    print(f"Total trainable parameters in the model.up: {count_parameters(model.branch2_up)}")
    print(f"Total trainable parameters in the model.down: {count_parameters(model.branch2_down)}")
    
    print(f"Total trainable parameters in the branch3: {count_parameters(model.branch3)}")

    # print(model)
    # create a input data
    input_data = torch.randn((4, 256, 64, 64)).cuda()
    output_data, output_image = model(input_data)
    print("input_data.shape: ", input_data.shape)
    print("output_data.shape: ", output_data.shape)
    print("output_image.shape: ", output_image.shape)