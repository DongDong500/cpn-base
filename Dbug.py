from torchsummary import summary
import torch

if __name__ == "__main__":
    import models

    net = models.models.__dict__['backbone_resnet50']()
    
    inputs = torch.rand(5, 3, 640, 640)
    print(summary(net, (3, 640, 640), device='cpu'))
    print(net(inputs).shape)
    #print(net.parameters())