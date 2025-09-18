import torch
from models.common import PP_LCNet

m = PP_LCNet(width_mult=1.0)     # 方案A：默认 (2,3,4)
x = torch.zeros(1,3,640,640)
c3, c4, c5 = m(x)
print(c3.shape, c4.shape, c5.shape)  # 约 [1,~64,80,80] [1,~160,40,40] [1,~320,20,20]
