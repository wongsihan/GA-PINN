import os
import time
import numpy as np
import torch
from torch.autograd import Variable, grad
from pyDOE import lhs
from scipy import integrate

# 设置GPU和随机种子
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 1234
torch.set_default_dtype(torch.float)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
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

def loss_grad_stats(loss, net):
    """
    功能：提供损失函数反向传播梯度的标准差和峰度
    输入：loss: 损失函数; net: 神经网络模型
    输出：标准差和峰度
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    # 收集梯度
    for m in net.modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        if (m == 0):
            w = grad(loss, m.weight, retain_graph=True)[0]
            b = grad(loss, m.bias, retain_graph=True)[0]
            grad_ = torch.cat((w.view(-1), b))
        else:
            w = grad(loss, m.weight, retain_graph=True)[0]
            if grad(loss, m.bias, retain_graph=True, allow_unused=True)[0] == None:
                continue
            b = grad(loss, m.bias, retain_graph=True)[0]
            grad_ = torch.cat((grad_, w.view(-1), b))
    # 收集梯度统计信息
    mean = torch.mean(grad_)
    diffs = grad_ - mean
    std = torch.std(grad_)
    zscores = diffs / std
    kurtoses = torch.mean(torch.pow(zscores, 4.0))

    return std, kurtoses

def loss_grad_max_mean(loss, net, lambg=1):
    """
    功能：提供损失函数反向传播梯度的最大值和均值
    输入：loss: 损失函数; net: 神经网络模型; lambg: 加权统计项（可选）
    输出：最大值和均值

    此实现基于：Wang et al: https://arxiv.org/pdf/2001.04536.pdf
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    for m in net.modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        if (m == 0):
            w = torch.abs(lambg * grad(loss, m.weight, retain_graph=True)[0])
            b = torch.abs(lambg * grad(loss, m.bias, retain_graph=True)[0])
            grad_ = torch.cat((w.view(-1), b))
        else:
            w = torch.abs(lambg * grad(loss, m.weight, retain_graph=True)[0])
            if grad(loss, m.bias, retain_graph=True, allow_unused=True)[0] == None:
                continue
            b = torch.abs(lambg * grad(loss, m.bias, retain_graph=True)[0])
            grad_ = torch.cat((grad_, w.view(-1), b))
    maxgrad = torch.max(grad_)
    meangrad = torch.mean(grad_)
    return maxgrad, meangrad
