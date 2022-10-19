from torchsummary import summary
import torch

import models
from get_argparser import get_argparser

if __name__ == "__main__":
    args = get_argparser().parse_args()

    net = models.models.__dict__['medt'](args, )
    
    inputs = torch.rand(5, 3, 640, 640)
    print(summary(net, (3, 640, 640), device='cpu'))
    print(net(inputs).shape)
    #print(net.parameters())