import os
import torch
import numpy as np
from pyDOE import lhs

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 设置随机种子
seed = 1234
torch.set_default_dtype(torch.float)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# GPU设置
use_gpu = torch.cuda.is_available()
print('GPU:', use_gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_fun(num, lb, ub):
    """利用拉丁超立方生成数据"""
    temp = torch.from_numpy(lb + (ub - lb) * lhs(2, num)).float()
    if use_gpu:
        temp = temp.cuda()
    return temp

def is_cuda(data):
    """将数据移动到GPU"""
    if use_gpu:
        data = data.cuda()
    return data

def tensor(x, **kw):
    """返回适合设备的张量"""
    return torch.tensor(x, dtype=torch.float32, **kw)

def get_device():
    """返回当前设备"""
    return device


