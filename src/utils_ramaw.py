import os
import torch
import numpy as np
from pyDOE import lhs

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 1234
torch.set_default_dtype(torch.float)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# 设备配置
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_cuda(data):
    """检查并移动数据到GPU"""
    if use_gpu:
        data = data.cuda()
    return data

def tensor(data, requires_grad=True):
    """创建张量并设置梯度要求"""
    t = torch.tensor(data, dtype=torch.float32, requires_grad=requires_grad)
    return is_cuda(t)

def random_fun(num, lb, ub):
    """利用拉丁超立方生成数据"""
    temp = torch.from_numpy(lb + (ub - lb) * lhs(2, num)).float()
    return is_cuda(temp)

def get_device():
    """返回当前设备"""
    return device

