from torch.nn import MSELoss

try:
    from .dice import DiceLoss
    from .entropydice import EntropyDiceLoss
    from .crossentropy import CrossEntropyLoss
except:
    from dice import DiceLoss
    from entropydice import EntropyDiceLoss
    from crossentropy import CrossEntropyLoss

def entropydice(**kwargs):

    return EntropyDiceLoss(update_weight=False)

def dice(**kwargs):

    return DiceLoss()

def crossentropy(**kwargs):

    return CrossEntropyLoss()

def mseloss(**kwargs):

    return MSELoss()

if __name__ == "__main__":

    import torch
    import torch.nn as nn
    
    loss = mseloss()

    m = nn.Softmax()

    s_input = torch.randn((16, 2), requires_grad=True)
    target = torch.randn((16, 2), requires_grad=True)
    # t_input = torch.rand((5, 2, 256, 256), requires_grad=True)
    #target = torch.randint(0, 2, (5, 256, 256))

    print("s_input: \t{}".format(s_input.size()))
    # print("t_input: \t{}".format(t_input.size()))
    print("target: \t{}".format(target.size()))
    print(loss(s_input, target))
    print(s_input, '\n', target)