import torch
import torch.nn as nn

def doubconv(in_c,out_c):
    cn = nn.Sequential(
        nn.Conv2d(in_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return cn

def crop_img(tensor,target_tensor):
    tensor_size=tensor.size()[2]
    target_size=target_tensor.size()[2]
    diff=tensor_size-target_size
    diff=diff//2
    return tensor[:,:,diff:tensor_size-diff,diff:tensor_size-diff]


class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downconv1 = doubconv(1,64)
        self.downconv2 = doubconv(64,128)
        self.downconv3 = doubconv(128,256)
        self.downconv4 = doubconv(256,512)
        self.downconv5 = doubconv(512,1024)

        self.up_trans1 = nn.ConvTranspose2d(in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2)


        self.up_conv1=doubconv(1024,512)

        self.up_trans2 = nn.ConvTranspose2d(in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2)


        self.up_conv2=doubconv(512,256)

        self.up_trans3 = nn.ConvTranspose2d(in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2)


        self.up_conv3=doubconv(256,128)

        self.up_trans4 = nn.ConvTranspose2d(in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2)


        self.up_conv4=doubconv(128,64)

        self.out=nn.Conv2d(in_channels=64,out_channels=2,kernel_size=1)

    def forward(self,image):
        #encoder down sampling
        x1=self.downconv1(image)
        #print(x1.size())
        x2=self.max_pool_2(x1)
        x3=self.downconv2(x2)
        x4=self.max_pool_2(x3)
        x5=self.downconv3(x4)
        x6=self.max_pool_2(x5)
        x7=self.downconv4(x6)
        x8=self.max_pool_2(x7)
        x9=self.downconv5(x8)


        #decoder up sampling
        x=self.up_trans1(x9)
        y=crop_img(x7,x)
        x=self.up_conv1(torch.cat([x,y],1))

        x=self.up_trans2(x)
        y=crop_img(x5,x)
        x=self.up_conv2(torch.cat([x,y],1))

        x=self.up_trans3(x)
        y=crop_img(x3,x)
        x=self.up_conv3(torch.cat([x,y],1))

        x=self.up_trans4(x)
        y=crop_img(x1,x)
        x=self.up_conv4(torch.cat([x,y],1))
        
        x=self.out(x)
        print(x.size())
        return x
        
        
        

if __name__ == '__main__':
    image = torch.rand((1,1,572,572))
    model = Unet()
    print(model(image))

         
    
