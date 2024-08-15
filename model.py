import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel,kernel_size =3, stride = 1, padding = 1, bias = False):
        super(DoubleConv, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias = bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding, bias = bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True)

        )
    def forward(self, x):
        return self.conv2d(x)    


class UNET(nn.Module):
    def __init__(self, in_channel = 3, out_channel =1, features = [64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        for feature in features:
            self.downs.append(DoubleConv(in_channel, feature))
            in_channel = feature

        for feature in reversed(features):
            self.ups.append( 
                  nn.ConvTranspose2d(feature*2, feature, kernel_size = 2, stride = 2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], 2*features[-1]) 
        self.out_conv = nn.Conv2d(features[0], out_channel, kernel_size = 1)

    def forward(self, x):
            skip_connections = []
            for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)   

            x = self.bottleneck(x)

            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                skip = skip_connections.pop()    
                if x.shape != skip.shape:
                     x = TF.resize(x, size = skip.shape[2:])
                concat_skip = torch.cat((x, skip), dim = 1)
                x = self.ups[idx+1](concat_skip)

            return self.out_conv(x)         


# to test model correctly working or not

def test():
    x = torch.randn((3, 1, 165, 165))
    model = UNET(in_channel = 1, out_channel = 1)
    preds = model(x)
    print(preds.shape, x.shape)

if __name__ == "__main__":
    test()  