import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable, grad
from utils_pmc import is_cuda, tensor
from config_pmc import br, Pi, u0, alpha, beta

# 计算A1区域损失的函数
def x_f_loss_fun_A1(x, train_U):
    u = train_U(x)      # 对应论文中的A
    Adx = u[:, 0]  # A1
    Ady = u[:, 1]
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    # inner loss，beta是啥？2023年9月12日18:58:21，----来自Lubin论文，2023年9月13日08:53:27
    # Lubin论文公式2
    exact_sol_A1 = exact_A1(x1, x2)
    # 简化损失函数，直接比较预测值和解析解
    err1 = exact_sol_A1 - Adx
    err2 = exact_sol_A1 - Ady
    return (err1**2 + err2**2).mean()

# 计算A2区域损失的函数
def x_f_loss_fun_A2(x, train_U):
    u = train_U(x)      # 对应论文中的A
    Adx = u[:, 0]  # A2
    Ady = u[:, 1]
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    # inner loss，beta是啥？2023年9月12日18:58:21，----来自Lubin论文，2023年9月13日08:53:27
    # Lubin论文公式2
    exact_sol_A2 = exact_A2(x1, x2)
    # 简化损失函数，直接比较预测值和解析解
    err1 = exact_sol_A2 - Adx
    err2 = exact_sol_A2 - Ady
    return (err1**2 + err2**2).mean()

# 边界条件损失函数
def loss_b1(x, train_U):
    u = train_U(x)
    Adx = u[:, 0]
    Ady = u[:, 1]
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    exact_sol = exact_A1(x1, x2)
    # 简化损失函数，直接比较预测值和解析解
    err1 = exact_sol - Adx
    err2 = exact_sol - Ady
    return (err1**2 + err2**2).mean()

def loss_b2(x, train_U):
    u = train_U(x)
    Adx = u[:, 0]
    Ady = u[:, 1]
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    exact_sol = exact_A2(x1, x2)
    # 简化损失函数，直接比较预测值和解析解
    err1 = exact_sol - Adx
    err2 = exact_sol - Ady
    return (err1**2 + err2**2).mean()

def loss_b3(x, train_U):
    u = train_U(x)
    Adx = u[:, 0]
    Ady = u[:, 1]
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    exact_sol = exact_A2(x1, x2)
    # 简化损失函数，直接比较预测值和解析解
    err1 = exact_sol - Adx
    err2 = exact_sol - Ady
    return (err1**2 + err2**2).mean()

def loss_b4(x, train_U):
    u = train_U(x)
    Adx = u[:, 0]
    Ady = u[:, 1]
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    exact_sol = exact_A1(x1, x2)
    # 简化损失函数，直接比较预测值和解析解
    err1 = exact_sol - Adx
    err2 = exact_sol - Ady
    return (err1**2 + err2**2).mean()

def loss_b57(x1, x2, train_U):
    u1 = train_U(x1)
    u2 = train_U(x2)
    Adx1 = u1[:, 0]
    Ady1 = u1[:, 1]
    Adx2 = u2[:, 0]
    Ady2 = u2[:, 1]
    err1 = Adx1 - Adx2
    err2 = Ady1 - Ady2
    return (err1**2 + err2**2).mean()

def loss_b68(x1, x2, train_U):
    u1 = train_U(x1)
    u2 = train_U(x2)
    Adx1 = u1[:, 0]
    Ady1 = u1[:, 1]
    Adx2 = u2[:, 0]
    Ady2 = u2[:, 1]
    err1 = Adx1 - Adx2
    err2 = Ady1 - Ady2
    return (err1**2 + err2**2).mean()

# 解析解函数
def exact_A1(x1, x2):
    from config_pmc import br, Pi, u0, alpha, beta
    mztest = ((4 * br) / (Pi * u0)) * (torch.sin(tensor(alpha * Pi / 2)))*(torch.cos(-1*beta*x1+(Pi/2)))
    return mztest

def exact_A2(x1, x2):
    from config_pmc import beta, K, b, c, d
    mztest = K * (torch.exp(-beta * x2) * torch.cos(beta * x1 + b) + c * torch.exp(beta * x2) * torch.cos(beta * x1 + d))
    return mztest
