import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self, in_channels=19, out_channels=19, features=64): # TODO: 코드에서는 features=32이다. 문제있으면 돌리기.
        super(UNet, self).__init__()

        self.encoder1 = UNet.double_conv(in_channels, features, kernel_size=3, padding=1) # TODO: 왜 self.double_conv가 아니라 UNet.double_conv일까
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet.double_conv(features, features*2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet.double_conv(features*2, features*4, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet.double_conv(features*4, features*8, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.middle = UNet.double_conv(features*8, features*16, kernel_size=3, padding=1)
        
        self.upconv4 = nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder4 = UNet.double_conv(features*16,features*8, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = UNet.double_conv(features*8, features*4, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = UNet.double_conv(features*4, features*2, kernel_size=3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet.double_conv(features*2, features, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(features, out_channels, kernel_size=1)
    
    
    def double_conv(in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True), # TODO: bias true,false는 어떤 영향을 줄까?
            # nn.BatchNorm2d(num_features=out_channels),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True),
            # nn.BatchNorm2d(num_features=out_channels),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
            
            
    def forward(self,x):
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)
        enc3 = self.encoder3(pool2)
        pool3 = self.pool3(enc3)
        enc4 = self.encoder4(pool3)
        pool4 = self.pool4(enc4)
        
        middle = self.middle(pool4)
        
        upconv4 = self.upconv4(middle)
        upconv4 = torch.cat((enc4,upconv4), dim=1)
        dec4 = self.decoder4(upconv4)
        upconv3 = self.upconv3(dec4)
        upconv3 = torch.cat((enc3,upconv3), dim=1)
        dec3 = self.decoder3(upconv3)
        upconv2 = self.upconv2(dec3)
        upconv2 = torch.cat((enc2,upconv2), dim=1)
        dec2 = self.decoder2(upconv2)
        upconv1 = self.upconv1(dec2)
        upconv1 = torch.cat((enc1,upconv1), dim=1)
        dec1 = self.decoder1(upconv1)
        output = torch.relu(self.conv1(dec1)) 
        # output = torch.sigmoid(self.conv1(dec1)) 
        # output = self.conv1(dec1)
        # output = torch.tanh(self.conv1(dec1)) 
        return output 