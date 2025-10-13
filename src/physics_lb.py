import torch
import numpy as np
from torch.autograd import Variable, grad
from utils_lb import is_cuda, tensor

# 计算A1区域损失的函数
def x_f_loss_fun_A1(x, train_U):
    if not x.requires_grad:
        x = Variable(x, requires_grad=True)
    u = train_U(x)      # 对应论文中的A
    Adx = u[:, 0]  # A1
    Ady = u[:, 1]
    d1 = torch.autograd.grad(Adx, x, grad_outputs=torch.ones_like(Adx), create_graph=True)
    u_x1x1 = d1[0][:, 0].unsqueeze(-1)
    d2 = torch.autograd.grad(Ady, x, grad_outputs=torch.ones_like(Ady), create_graph=True)
    u_x2x2 = d2[0][:, 1].unsqueeze(-1)
    #u_x2 = d[0][:, 1].unsqueeze(-1)
    # u_x1x1 = torch.autograd.grad(u_x1, x, grad_outputs=torch.ones_like(u_x1), create_graph=True)[0][:, 0].unsqueeze(-1)
    # u_x2x2 = torch.autograd.grad(u_x2, x, grad_outputs=torch.ones_like(u_x2), create_graph=True)[0][:, 1].unsqueeze(-1)
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    # inner loss，beta是啥？2023年9月12日18:58:21，----来自Lubin论文，2023年9月13日08:53:27
    # Lubin论文公式2
    exact_sol_A1 = exact_A1(x1, x2)
    dif_exact_sol1 = grad(exact_sol_A1, (x1,x2), grad_outputs=torch.ones_like(exact_sol_A1), create_graph=True)
    err1 = dif_exact_sol1[0].squeeze(-1) - Adx
    err2 = dif_exact_sol1[1].squeeze(-1) - Ady
    #Mzdx = (4 * br / Pi / u0) * (torch.sin(tensor(alpha * Pi / 2))) * beta * (torch.cos(beta * x1))
    mztest = ((4 * br) / (Pi * u0)) * (torch.sin(tensor(alpha * Pi / 2)))*(torch.cos(-1*beta*x1+(Pi/2)))
    # for n in range(2, 3):
    #     mztest = mztest + ((4 * br) / ((2 * n - 1) * Pi * u0)) * (torch.sin(tensor((2 * n - 1) * alpha * Pi / 2))) * (
    #         torch.cos((-1) * (2 * n - 1) * beta * x1 + (2 * n - 1) * (Pi / 2)))

    mztest = grad(mztest, x1, grad_outputs=torch.ones_like(mztest), create_graph=True)[0]
    # mz = (4 * br / (Pi * u0)) * (torch.sin(tensor(alpha * Pi / 2))) * (torch.cos(tensor(-1 * beta) * x1 + (Pi / 2)))#后面这个是从lubbin转换成wangjian的
    # for n in range(2, 3):
    #     mz = mz + ((4 * br) / ((2 * n - 1) * Pi * u0)) * (torch.sin(tensor((2 * n - 1) * alpha * Pi / 2))) * (
    #         torch.cos((-1) * (2 * n - 1) * beta * x1 + (2 * n - 1) * (Pi / 2)))
    # Mzdx = grad(mz, x1, grad_outputs=torch.ones_like(mz), create_graph=True)[0]

    f = u_x1x1 + u_x2x2 + u0 * mztest     # 永磁体区域的损失函数，由Lubin论文公式（1）推导而来
    # 边界1,5,7的损失也在这里计算，最后把边界损失计算均方差后相加
    # 残差
    return  ((f)**2)

def x_f_loss_fun_A2(x, train_U):
    if not x.requires_grad:
        x = Variable(x, requires_grad=True)
    x = is_cuda(x)

    u = train_U(x)
    Adx = u[:, 0]  # A1
    Ady = u[:, 1]
    d1 = torch.autograd.grad(Adx, x, grad_outputs=torch.ones_like(Adx), create_graph=True)
    u_x1x1 = d1[0][:, 0].unsqueeze(-1)
    d2 = torch.autograd.grad(Ady, x, grad_outputs=torch.ones_like(Ady), create_graph=True)
    u_x2x2 = d2[0][:, 1].unsqueeze(-1)
    # u_x1x1 = torch.autograd.grad(u_x1, x, grad_outputs=torch.ones_like(u_x1), create_graph=True)[0][:, 0].unsqueeze(-1)
    # u_x2x2 = torch.autograd.grad(u_x2, x, grad_outputs=torch.ones_like(u_x2), create_graph=True)[0][:, 1].unsqueeze(-1)
    x1 = x[:, [0]]
    x2 = x[:, [1]]

    #inner loss
    #Mzdx = (4 * br / Pi * u0) * (torch.sin(alpha * Pi / 2)) * beta * (torch.cos(beta * x1))
    f = u_x1x1 + u_x2x2
    exact_sol_A2 = exact_A2(x1, x2)
    dif_exact_sol1 = grad(exact_sol_A2, (x1, x2), grad_outputs=torch.ones_like(exact_sol_A2), create_graph=True)
    err1 = dif_exact_sol1[0].squeeze(-1) - Adx
    err2 = dif_exact_sol1[1].squeeze(-1) - Ady

    return ((f) ** 2)

# 这里的x必须带边界的坐标进去
def loss_b1(x, train_U):
    if not x.requires_grad:
        x = Variable(x, requires_grad=True)
    x = is_cuda(x)
    u = train_U(x)      # 对应论文中的A
    #d = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
    Adx = u[:, 0].unsqueeze(-1)
    Ady = u[:, 1].unsqueeze(-1)
    #my_loss_b1 = Ady
    d1 = torch.autograd.grad(Adx, x, grad_outputs=torch.ones_like(Adx), create_graph=True)
    u_x1x1 = d1[0][:, 0].unsqueeze(-1)
    d2 = torch.autograd.grad(Ady, x, grad_outputs=torch.ones_like(Ady), create_graph=True)
    u_x2x2 = d2[0][:, 1].unsqueeze(-1)
    x1 = x[:,[0]]
    x2 = x[:,[1]]
    exact_sol1 = exact_A2(x1, x2)
    dif_exact_sol1 = grad(exact_sol1, (x1,x2), grad_outputs=torch.ones_like(exact_sol1), create_graph=True)
    err1 = dif_exact_sol1[0] - Adx
    err2 = dif_exact_sol1[1] - Ady
    err3 = u_x1x1 + u_x2x2
    # loss = ((my_loss_b1)**2).sum() + ((err1)**2).sum()
    return torch.mean((err1)**2, dim=0) + torch.mean((err2)**2, dim=0)

def loss_b2(x, train_U):
    if not x.requires_grad:
        x = Variable(x, requires_grad=True)
    x = is_cuda(x)
    u = train_U(x)      # 对应论文中的A
    #d = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
    Adx = u[:, 0].unsqueeze(-1)
    Ady = u[:, 1].unsqueeze(-1)
    d1 = torch.autograd.grad(Adx, x, grad_outputs=torch.ones_like(Adx), create_graph=True)
    u_x1x1 = d1[0][:, 0].unsqueeze(-1)
    d2 = torch.autograd.grad(Ady, x, grad_outputs=torch.ones_like(Ady), create_graph=True)
    u_x2x2 = d2[0][:, 1].unsqueeze(-1)

    x1 = x[:,[0]]
    x2 = x[:,[1]]
    exact_sol1 = exact_A2(x1, x2)
    dif_exact_sol1 = grad(exact_sol1, (x1,x2), grad_outputs=torch.ones_like(exact_sol1), create_graph=True)
    err1 = dif_exact_sol1[0] - Adx
    err2 = dif_exact_sol1[1] - Ady
    err3 = u_x1x1 + u_x2x2
    return torch.mean((err1)**2, dim=0) + torch.mean((err2)**2, dim=0)

def loss_b3(x, train_U):
    if not x.requires_grad:
        x = Variable(x, requires_grad=True)
    x = is_cuda(x)
    u = train_U(x)  # 对应论文中的A
    # d = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
    Adx = u[:, 0].unsqueeze(-1)
    Ady = u[:, 1].unsqueeze(-1)
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    d1 = torch.autograd.grad(Adx, x, grad_outputs=torch.ones_like(Adx), create_graph=True)
    u_x1x1 = d1[0][:, 0].unsqueeze(-1)
    d2 = torch.autograd.grad(Ady, x, grad_outputs=torch.ones_like(Ady), create_graph=True)
    u_x2x2 = d2[0][:, 1].unsqueeze(-1)

    exact_sol1 = exact_A1(x1, x2)
    dif_exact_sol1 = grad(exact_sol1, (x1, x2), grad_outputs=torch.ones_like(exact_sol1), create_graph=True)
    err1 = dif_exact_sol1[0] - Adx
    err2 = dif_exact_sol1[1] - Ady

    exact_sol2 = exact_A2(x1, x2)
    dif_exact_sol2 = grad(exact_sol2, (x1, x2), grad_outputs=torch.ones_like(exact_sol2), create_graph=True)
    err3 = dif_exact_sol2[0] - Adx
    err4 = dif_exact_sol2[0] - Ady

    Mzdx = (4 * br / Pi / u0) * (torch.sin(tensor(alpha * Pi / 2))) * beta * (torch.cos(beta * x1))
    err5 = u_x1x1 + u_x2x2 + u0 * Mzdx
    err6 = u_x1x1 + u_x2x2
    return torch.mean((err1) ** 2, dim=0)+torch.mean((err2) ** 2, dim=0)+torch.mean((err3) ** 2, dim=0)+torch.mean((err4) ** 2, dim=0)

def loss_b4(x, train_U):
    if not x.requires_grad:
        x = Variable(x, requires_grad=True)
    x = is_cuda(x)
    u = train_U(x)      # 对应论文中的A
    #d = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
    Adx = u[:, 0].unsqueeze(-1)
    Ady = u[:, 1].unsqueeze(-1)
    #my_loss_b4 = Ady

    x1 = x[:, [0]]
    x2 = x[:, [1]]
    d1 = torch.autograd.grad(Adx, x, grad_outputs=torch.ones_like(Adx), create_graph=True)
    u_x1x1 = d1[0][:, 0].unsqueeze(-1)
    d2 = torch.autograd.grad(Ady, x, grad_outputs=torch.ones_like(Ady), create_graph=True)
    u_x2x2 = d2[0][:, 1].unsqueeze(-1)

    exact_sol4 = exact_A1(x1, x2)
    dif_exact_sol4 = grad(exact_sol4, (x1, x2), grad_outputs=torch.ones_like(exact_sol4), create_graph=True)
    err4 = dif_exact_sol4[0] - Adx
    err5 = dif_exact_sol4[1] - Ady

    #Mzdx = (4 * br / Pi / u0) * (torch.sin(tensor(alpha * Pi / 2))) * beta * (torch.cos(beta * x1))
    mztest = ((4 * br) / (Pi * u0)) * (torch.sin(tensor(alpha * Pi / 2))) * (torch.cos(-1 * beta * x1 + (Pi / 2)))
    # for n in range(2, 3):
    #     mztest = mztest + ((4 * br) / ((2 * n - 1) * Pi * u0)) * (torch.sin(tensor((2 * n - 1) * alpha * Pi / 2))) * (
    #         torch.cos((-1) * (2 * n - 1) * beta * x1 + (2 * n - 1) * (Pi / 2)))

    mztest = grad(mztest, x1, grad_outputs=torch.ones_like(mztest), create_graph=True)[0]
    err6 = u_x1x1 + u_x2x2 + u0 * mztest
    #loss = ((my_loss_b4)**2).sum() + ((err4)**2).sum()
    return torch.mean((err4)**2, dim=0)+torch.mean((err5)**2, dim=0)

def loss_b57(x1,x2, train_U):
    if not x1.requires_grad:
        x1 = Variable(x1, requires_grad=True)
    if not x2.requires_grad:
        x2 = Variable(x2, requires_grad=True)
    x1 = is_cuda(x1)
    x2 = is_cuda(x2)

    u1 = train_U(x1)      # 对应论文中的A
    u2 = train_U(x2)      # 对应论文中的A

    A5dy = u1[:, 1].unsqueeze(-1)
    A5dx = u1[:, 0].unsqueeze(-1)
    A7dy = u2[:, 1].unsqueeze(-1)
    A7dx = u2[:, 0].unsqueeze(-1)
    x51 = x1[:, [0]]
    x52 = x1[:, [1]]
    x71 = x2[:, [0]]
    x72 = x2[:, [1]]
    exact_sol5 = exact_A2(x51, x52)
    dif_exact_sol5 = grad(exact_sol5, (x51, x52), grad_outputs=torch.ones_like(exact_sol5), create_graph=True)
    err5 = dif_exact_sol5[0] - A5dx
    err6 = dif_exact_sol5[1] - A5dy

    exact_sol7 = exact_A2(x71, x72)
    dif_exact_sol7 = grad(exact_sol7, (x71, x72), grad_outputs=torch.ones_like(exact_sol7), create_graph=True)
    err7 = dif_exact_sol7[0] - A7dx
    err8 = dif_exact_sol7[1] - A7dy

    err9 = A5dx - A7dx

    return torch.mean((err5)**2, dim=0)+torch.mean((err6)**2, dim=0)+torch.mean((err7)**2, dim=0)+torch.mean((err8)**2, dim=0)+torch.mean((err9)**2, dim=0)

def loss_b68(x1, x2, train_U):
    if not x1.requires_grad:
        x1 = Variable(x1, requires_grad=True)
    if not x2.requires_grad:
        x2 = Variable(x2, requires_grad=True)

    x1 = is_cuda(x1)
    x2 = is_cuda(x2)

    u1 = train_U(x1)  # 对应论文中的A
    u2 = train_U(x2)  # 对应论文中的A

    A6dy = u1[:, 1].unsqueeze(-1)
    A6dx = u1[:, 0].unsqueeze(-1)
    A8dy = u2[:, 1].unsqueeze(-1)
    A8dx = u2[:, 0].unsqueeze(-1)

    x61 = x1[:, [0]]
    x62 = x1[:, [1]]
    x81 = x2[:, [0]]
    x82 = x2[:, [1]]
    exact_sol6 = exact_A1(x61, x62)
    dif_exact_sol6 = grad(exact_sol6, (x61, x62), grad_outputs=torch.ones_like(exact_sol6), create_graph=True)
    err6 = dif_exact_sol6[0] - A6dx
    err7 = dif_exact_sol6[1] - A6dy

    exact_sol8 = exact_A1(x81, x82)
    dif_exact_sol8 = grad(exact_sol8, (x81, x82), grad_outputs=torch.ones_like(exact_sol8), create_graph=True)
    err8 = dif_exact_sol8[0] - A8dx
    err9 = dif_exact_sol8[1] - A8dy

    return torch.mean((err6)**2, dim=0) + torch.mean((err7)**2, dim=0)+ torch.mean((err8)**2, dim=0) + torch.mean((err9)**2, dim=0)

def exact_A1(x,y):
    # res = K * (1 - ((torch.sinh(tensor(beta * (c + d), requires_grad=True)) * (torch.cosh(tensor(beta, requires_grad=True)*y).unsqueeze(-1))) / torch.sinh(
    #     tensor(beta * (b + c + d), requires_grad=True)))) * (torch.cos(tensor(beta, requires_grad=True)*x).unsqueeze(-1))
    exact_A1 = 1 - ((torch.sinh(tensor(beta * (c + d), requires_grad=True)) * torch.cosh(tensor(beta, requires_grad=True) * y)) / torch.sinh(
        tensor(beta * (b + c + d), requires_grad=True)))
    exact_A1 = K * exact_A1 * torch.cos(tensor(beta, requires_grad=True) * x)
    # res = grad(
    #         outputs=exact_A1, inputs=(x, y), grad_outputs=torch.ones_like(exact_A1),
    #         create_graph=True
    #     )
    # res_dx = res[0]
    # res_dy = res[1]
    return exact_A1

def exact_A2(x,y):
    # res = K * (torch.sinh(tensor(beta * b, requires_grad=True)) * (torch.cosh(tensor(beta, requires_grad=True)*(y - (b + c + d))).unsqueeze(-1)) * (torch.cos(
    #     tensor(beta, requires_grad=True)*x).unsqueeze(-1))) / torch.sinh(tensor(beta * (b + c + d), requires_grad=True))
    exact_A2 = K * (torch.sinh(tensor(beta * b, requires_grad=True)) * torch.cosh(tensor(beta, requires_grad=True) * (y - (b + c + d))) * torch.cos(
        tensor(beta, requires_grad=True) * x))
    exact_A2 = exact_A2 / torch.sinh(tensor(beta * (b + c + d), requires_grad=True))
    # res = grad(
    #     outputs=exact_A2, inputs=(x, y), grad_outputs=torch.ones_like(exact_A2),
    #     create_graph=True
    # )
    # res_dx = res[0]
    # res_dy = res[1]
    return exact_A2


