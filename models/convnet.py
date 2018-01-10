import torch
from torch import nn

dropout = torch.nn.Dropout(p=0.20)

def make_linear_bn_relu_dropout(in_channels, out_channels, p):
    return [
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(p)
    ]

class ConvRes(nn.Module):
    def __init__(self,insize, outsize):
        super(ConvRes, self).__init__()
        drate = .3
        self.math = nn.Sequential(
                 nn.BatchNorm2d(insize),
                 nn.Dropout(drate),
                 torch.nn.Conv2d(insize, outsize, kernel_size=2,padding=2),
                 nn.PReLU(),
                )
        
    def forward(self, x):
        return self.math(x) 

class Conv_BN_ReLU(nn.Module):
    def __init__(self,insize, outsize, kernel_size=3, padding=1, groups=1):
        super(Conv_BN_ReLU, self).__init__()
        self.math = torch.nn.Sequential(
            torch.nn.Conv2d(insize, outsize, kernel_size=kernel_size,padding=padding, groups=groups),
            torch.nn.BatchNorm2d(outsize),
            torch.nn.ReLU(),
        )
        
    def forward(self, x):
        x=self.math(x)
        return x

class Conv_BN_ReLU_MaxPool(nn.Module):
    def __init__(self,insize, outsize, kernel_size=3, padding=1, groups=1, pool=2, stride=2):
        super(Conv_BN_ReLU_MaxPool, self).__init__()
        self.math = torch.nn.Sequential(
            torch.nn.Conv2d(insize, outsize, kernel_size=kernel_size,padding=padding, groups=groups),
            torch.nn.BatchNorm2d(outsize),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(pool,stride),
        )
        
    def forward(self, x):
        x=self.math(x)
        return x
        
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.cnn1 = Conv_BN_ReLU_MaxPool(3,64*3, kernel_size=3, padding=0, groups=3, pool=2)
        
        self.cnn2 = Conv_BN_ReLU_MaxPool(64*3,128*3, kernel_size=3,padding=0, groups=3, pool=2)
        
        self.cnn3 = Conv_BN_ReLU_MaxPool(128*3,128*3, kernel_size=3, padding=0, groups=3, pool=2)
        
        self.cnn4 = Conv_BN_ReLU_MaxPool(128*3,64*3, kernel_size=3, padding=0, groups=3, pool=2)
        
        self.features = nn.Sequential( 
            self.cnn1,dropout,          
            self.cnn2,dropout,
            self.cnn3,dropout,
            self.cnn4,dropout,
        )
        
        self.fc1 = nn.Sequential(*make_linear_bn_relu_dropout(256*3, 1024, 0.2))
        self.fc2 = nn.Sequential(*make_linear_bn_relu_dropout(1024, 1024, 0.2))
        self.fc3 = nn.Sequential(*make_linear_bn_relu_dropout(1024, 256, 0.2))
        
        self.classifier = torch.nn.Sequential(
            nn.Linear(256, 1),             
        )
        self.sig=nn.Sigmoid()        
            
    def forward(self, x):
        x = self.features(x) 
        x = x.view(x.size(0), -1)    
        x = self.fc1(x)
        x = self.fc2(x)  
        x = self.fc3(x)  
        x = self.classifier(x)                
        x = self.sig(x)
        return x  
