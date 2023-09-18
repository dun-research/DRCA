
import torch.nn.functional as F
from torch import nn
from torch.nn import Dropout
from torch.nn import init

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''

    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, (nn.Linear, nn.Embedding, nn.Conv1d)):
        #m.weight.data.normal_(mean=0.0, std=0.1)
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LayerNorm):
        m.bias.data.zero_()
        m.weight.data.fill_(1.0)