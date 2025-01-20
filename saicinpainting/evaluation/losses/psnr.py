import torch
import torch.nn as nn
import numpy as np
# 1. img得是255
# 2、b h, w, c
# 3、int格式
class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean(torch.tensor((a.astype(np.float32) - b.astype(np.float32)) ** 2))

        if mse == 0:
            return 0

        return self.max_val - 10 * torch.log(mse) / self.base10