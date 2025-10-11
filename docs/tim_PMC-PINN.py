import os
import time
import numpy as np
import torch
from matplotlib import pyplot as plt, gridspec
from torch import nn
from torch.autograd import Variable, grad
from tqdm import tqdm, trange
from pyDOE import lhs
from scipy import integrate



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 1234
torch.set_default_dtype(torch.float)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
use_gpu = torch.cuda.is_available()
print('GPU:', use_gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 利用拉丁超立方生成数据
def  random_fun(num, lb, ub):
    temp = torch.from_numpy(lb + (ub - lb) * lhs(2, num)).float()
    if use_gpu:
        temp = temp.cuda()
    return temp

def is_cuda(data):
    if use_gpu:
        data = data.cuda()
    return data

def loss_grad_stats(loss, net):
    """
    Functionality: provides std, kurtosis of backpropagated gradients of loss function
    inputs: loss: loss function ; net: the NN model
    outputs: std and kurtosis
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    # collect gradients
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        if (m == 0):
            w = grad(loss, m.weight, retain_graph=True)[0]
            b = grad(loss, m.bias, retain_graph=True)[0]
            grad_ = torch.cat((w.view(-1), b))
        else:
            w = grad(loss, m.weight, retain_graph=True)[0]
            if grad(loss, m.bias, retain_graph=True,allow_unused=True)[0] == None:
                continue
            b = grad(loss, m.bias, retain_graph=True)[0]
            grad_ = torch.cat((grad_, w.view(-1), b))
    # collect gradient statistics
    mean = torch.mean(grad_)
    diffs = grad_ - mean
    # var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.std(grad_)
    zscores = diffs / std
    # skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0))

    return std, kurtoses

def loss_grad_max_mean(loss, net, lambg=1):
    """
    Functionality: provides maximum and mean of backpropagated gradients of loss function
    inputs: loss: loss function ; net: the NN model; lambg : term for weighted stats (optional)
    outputs: max and mean

    This implementation is according to: Wang et al: https://arxiv.org/pdf/2001.04536.pdf
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        if (m == 0):
            w = torch.abs(lambg * grad(loss, m.weight, retain_graph=True)[0])
            b = torch.abs(lambg * grad(loss, m.bias, retain_graph=True)[0])
            grad_ = torch.cat((w.view(-1), b))
        else:
            w = torch.abs(lambg * grad(loss, m.weight, retain_graph=True)[0])
            if grad(loss, m.bias, retain_graph=True,allow_unused=True)[0] == None:
                continue
            b = torch.abs(lambg * grad(loss, m.bias, retain_graph=True)[0])
            grad_ = torch.cat((grad_, w.view(-1), b))
    maxgrad = torch.max(grad_)
    meangrad = torch.mean(grad_)
    return maxgrad, meangrad

# 全连接神经网络
class Net(nn.Module):
    def __init__(self, layers, mean=0, std=1):
        super(Net, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.mean = mean
        self.std  = std
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)
    # 前向传播
    def forward(self, x):
        x = (x-self.mean.to(device))/self.std.to(device)
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)     # (1500,2)
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
        a = self.linear[-1](a)
        return a

class Model:
    def __init__(self, net, b1, b2, b3, b4, b5, b7, b6, b8, loss_b1,loss_b2,loss_b3, loss_b4, loss_b57, loss_b68, x_f_loss_fun_A1, x_f_loss_fun_A2,
                  x_test, x_test_exact_Adx,x_test_exact_Ady, x_f_N_A1, x_f_M_A1, x_f_N_A2, x_f_M_A2,method,x_test_copper,x_test_copper_Adx,
                 x_test_copper_Ady
                 ):
        self.method = method
        self.x_label_s = None
        self.x_f_s = None
        self.s_collect = []
        self.optimizer_LBGFS = None
        self.net = net
        # 在A1域内取的不动的点
        self.x_f_N_A1 = x_f_N_A1
        self.x_f_N_A2 = x_f_N_A2
        self.x_f_M_A1 = x_f_M_A1
        self.x_f_M_A2 = x_f_M_A2
        # 边界上的点
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.b5 = b5
        self.b7 = b7
        self.b6 = b6
        self.b8 = b8
        # 边界上的损失函数
        self.loss_b1 = loss_b1
        self.loss_b2 = loss_b2
        self.loss_b3 = loss_b3
        self.loss_b4 = loss_b4
        self.loss_b57 = loss_b57
        self.loss_b68 = loss_b68
        self.x_f_loss_fun_A1 = x_f_loss_fun_A1
        self.x_f_loss_fun_A2 = x_f_loss_fun_A2
        self.x_test = x_test
        self.x_test_exact_Adx = x_test_exact_Adx
        self.x_test_exact_Ady = x_test_exact_Ady
        self.start_loss_collect = False
        self.x_label_loss_collect = []
        self.x_f_loss_collect = []
        self.x_test_estimate_collect = []
        self.x_test_copper = x_test_copper
        self.x_test_copper_Adx = x_test_copper_Adx
        self.x_test_copper_Ady = x_test_copper_Ady

    def train_U(self, x):
        return self.net(x)

    def predict_U(self, x):
        return self.train_U(x)
    def cal_exact_torque(self,x1,x2):
        x_draw, y_draw = np.mgrid[length_left:length_right:field_res * 1j,
                         thickness_mag + thickness_air:thickness_mag + thickness_air + thickness_copper:field_res * 1j]

        xy_draw = np.c_[x_draw.ravel(), y_draw.ravel()]
        xy_draw = tensor(xy_draw, requires_grad=True)
        thisA2 = exact_A2(x1, x2)
        thisA2dx = grad(thisA2, x1, grad_outputs=torch.ones_like(thisA2), create_graph=True)
        thisA2dx = thisA2dx[0]
        thisres = thisA2dx.reshape(field_res, field_res)
        eddy_current = -sigma * omega * Rm * thisres
        J2 = eddy_current * eddy_current
        torque = integrate.simps(J2.T.cpu().detach(), y_draw.T, axis=0)
        torque = integrate.simps(torque, x_draw[:, 0])
        torque = (torque * 10 * L) / (sigma * omega)
        return torque
    # 画出涡流，磁感应强度和Adx
    def render_res(self, move_count, pic_name):
            # draw field
            # x_draw = random_fun(50, lb_copper, ub_copper)
            x_draw, y_draw = np.mgrid[length_left:length_right:field_res * 1j,
                             thickness_mag + thickness_air:thickness_mag + thickness_air + thickness_copper:field_res * 1j]
            # x_draw = torch.cat((self.x_f_N_A1, self.x_f_M_A1, self.x_f_N_A2, self.x_f_M_A2), dim=0)
            # if not x_draw.requires_grad:
            #     x_draw = Variable(x_draw, requires_grad=True)
            xy_draw = np.c_[x_draw.ravel(), y_draw.ravel()]
            xy_draw = tensor(xy_draw, requires_grad=True)
            # xy_draw2 = tensor(xy_draw, requires_grad=True)
            # 磁矢势A
            xy_draw = is_cuda(xy_draw)
            u = self.train_U(xy_draw)
            # MyConvNetVis = make_dot(u, params=dict(list(self.net.parameters())), show_attrs=True, show_saved=True)
            # MyConvNetVis.format = "png"
            # MyConvNetVis.directory = "data"
            # MyConvNetVis.view()

            # d = torch.autograd.grad(u, xy_draw, grad_outputs=torch.ones_like(u), create_graph=True)
            # u_x1 = d[0][:, 0].unsqueeze(-1)
            # u_x2 = d[0][:, 1].unsqueeze(-1)
            # u_x1x1 = torch.autograd.grad(u_x1, x_draw, grad_outputs=torch.ones_like(u_x1), create_graph=True)[0][:,
            #          0].unsqueeze(-1)
            # u_x2x2 = torch.autograd.grad(u_x2, x_draw, grad_outputs=torch.ones_like(u_x2), create_graph=True)[0][:,
            #          1].unsqueeze(-1)
            x1 = xy_draw[:, [0]]
            x2 = xy_draw[:, [1]]
            #
            # Adx =  u[:, 0]
            # Adx.shape = x_draw.shape +(2,)
            new_u = u.reshape(x_draw.shape + (2,))
            Adx = new_u[..., 0]
            Ady = new_u[..., 1]
            eddy_current = -sigma * omega * Rm * Adx
            # print("adx is :", Adx)
            # print("eddy_current is:", eddy_current)
            # 用于绘图
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # plt.plot(x1[:, [0]].detach().numpy(), x2[:, [0]].detach().numpy(),
            #          u_x1.detach().numpy().reshape(-1, 1))
            # plt.show()
            # eddy_current的数据格式为(250000,1),500*500个点的涡流数据
            J2 = eddy_current * eddy_current
            # dx = x_draw[:,0]
            # dy = y_draw.T
            # torque = integrate.simps(J2.reshape(field_res,field_res).detach().numpy(), dy, axis=0)
            # torque = (integrate.simps(torque, dx) * L * 10) / (sigma * omega)

            # torque = integrate.simps(J2.detach().numpy(),xy_draw[:,0].unsqueeze(-1).detach().numpy())
            # torque = integrate.simps(torque, xy_draw[:, 1].detach().numpy())

            # torque = integrate.simps(J2.reshape(field_res,field_res).detach().numpy(), x_draw[:,0])
            # torque = integrate.simps(torque, y_draw[0,:])
            # torque = np.trapz(J2.reshape(field_res, field_res).cpu().detach().numpy(), x_draw[:, 0], axis = 1)
            # torque = np.trapz(torque, y_draw[0, :])
            torque = integrate.simps(J2.T.cpu().detach(), y_draw.T, axis=0)
            torque = integrate.simps(torque, x_draw[:, 0])
            torque = (torque * 10 * L) / (sigma * omega)
            # print("Eddy current is:", eddy_current)
            print("Torque is:", torque)
            exact_torque = self.cal_exact_torque(x1, x2)
            torque_residual = abs(exact_torque - torque)
            print("torque_residual is:", torque_residual)
            # eddy_current = tensor(eddy_current, requires_grad=True)
            plt.figure(figsize=(20, 4))
            plt.subplot(1, 2, 1)
            plt.contour(x_draw, y_draw,
                        Adx[:, :].cpu().detach().numpy(), 10000, cmap='jet', zorder=1)
            plt.colorbar()
            plt.xlabel('x', fontsize=14, fontname='Times New Roman')
            plt.ylabel('y', fontsize=14, fontname='Times New Roman')
            # plt.title("eddy current")

            Bz = -Adx.cpu().detach().numpy()
            Bx = -Ady.cpu().detach().numpy()
            plt.subplot(1, 2, 2)
            M2 = np.hypot(Bx, Bz)
            # plt.quiver(-x_draw, y_draw, Bx, Bz, M2, width=0.005,
            #            scale=60, cmap='jet')
            plt.contour(x_draw, y_draw,
                        Ady[:, :].cpu().detach().numpy(), 10000, cmap='jet', zorder=1)
            #
            # plt.contour(xn_copper.cpu().detach().numpy(), yn_copper.cpu().detach().numpy(),
            #             eddy_current[:, :].cpu().detach().numpy(), 10000, cmap='jet', zorder=1)
            plt.colorbar()
            plt.xlabel('x', fontsize=14, fontname='Times New Roman')
            plt.ylabel('y', fontsize=14, fontname='Times New Roman')
            # plt.title("vector B")
            filename = f"{pic_name}1-{move_count}.png"
            filename2 = f"{pic_name}2-{move_count}.png"

            plt.title(filename)
            plt.savefig(filename)
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.plot(x1[:, [0]].cpu().detach().numpy(), x2[:, [0]].cpu().detach().numpy(),
                     u[:, 0].cpu().detach().numpy().reshape(-1, 1))

            plt.title(filename2)
            plt.savefig(filename2)
            plt.show()
            # plt.contour(xn_copper.cpu().detach().numpy(), yn_copper.cpu().detach().numpy(),
            #             eddy_current[:, :].cpu().detach().numpy(), 10000, cmap='jet', zorder=1)
            # plt.colorbar()
            # plt.xticks([])
            # plt.yticks([])
            # plt.title("vector B")
            # plt.show()

    def render_residual(self, move_count, pic_name='residual'):

        x_draw, y_draw = np.mgrid[length_left:length_right:field_res * 1j,
                         thickness_mag + thickness_air:thickness_mag + thickness_air + thickness_copper:field_res * 1j]

        xy_draw = np.c_[x_draw.ravel(), y_draw.ravel()]
        xy_draw = tensor(xy_draw, requires_grad=True)
        # xy_draw2 = tensor(xy_draw, requires_grad=True)
        # 磁矢势A
        xy_draw = is_cuda(xy_draw)
        u = self.train_U(xy_draw)#400，2

        x1 = xy_draw[:, [0]]#400，1
        x2 = xy_draw[:, [1]]

        thisA2 = exact_A2(x1, x2)#400，1
        thisA2dx = grad(thisA2, x1, grad_outputs=torch.ones_like(thisA2), create_graph=True)
        thisA2dx = thisA2dx[0]#400，1
        thisA2dy = grad(thisA2, x2, grad_outputs=torch.ones_like(thisA2), create_graph=True)
        thisA2dy = thisA2dy[0]
        Adx_exact = thisA2dx.reshape(field_res, field_res)#20,20
        Ady_exact = thisA2dy.reshape(field_res, field_res)

        # Adx =  u[:, 0]
        # Adx.shape = x_draw.shape +(2,)
        new_u = u.reshape(x_draw.shape + (2,))
        Adx_pred = new_u[..., 0]
        Ady_pred = new_u[..., 1]

        Adx = abs(Adx_exact-Adx_pred)
        Ady = abs(Ady_exact-Ady_pred)

        # eddy_current_pred = -sigma * omega * Rm * Adx_pred
        # eddy_current_exact = -sigma * omega * Rm * Adx_exact
        # eddy_current_res = abs(eddy_current_pred-eddy_current_exact)

        plt.figure(figsize=(20, 4))
        plt.subplot(1, 2, 1)
        plt.contourf(x_draw, y_draw,
                    Adx[:, :].cpu().detach().numpy(), 10000, cmap='plasma', zorder=1)
        plt.colorbar()
        plt.xlabel('x', fontsize=14, fontname='Times New Roman')
        plt.ylabel('y', fontsize=14, fontname='Times New Roman')

        # Bz_pred = -Adx_pred.cpu().detach().numpy()
        # Bx_pred = -Ady_pred.cpu().detach().numpy()
        # Bz_exact = -Adx_exact.cpu().detach().numpy()
        # Bx_exact = -Ady_exact.cpu().detach().numpy()
        # Bz_res = abs(Bz_pred - Bz_exact)
        # Bx_res = abs(Bx_pred - Bx_exact)
        # plt.subplot(1, 2, 2)
        # M2 = np.hypot(Bx_res, Bz_res)
        # plt.quiver(-x_draw, y_draw, Bx_res, Bz_res, M2, width=0.005,
        #            scale=30, cmap='jet')
        plt.subplot(1, 2, 2)
        plt.contourf(x_draw, y_draw,
                    Ady[:, :].cpu().detach().numpy(), 10000, cmap='plasma', zorder=1)
        plt.colorbar()
        plt.xlabel('x', fontsize=14, fontname='Times New Roman')
        plt.ylabel('y', fontsize=14, fontname='Times New Roman')
        filename = f"{pic_name}1-{move_count}-residual.png"
        filename2 = f"{pic_name}2-{move_count}-residual.png"

        plt.title(filename)
        plt.savefig(filename)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.plot(x1[:, [0]].cpu().detach().numpy(), x2[:, [0]].cpu().detach().numpy(),
                 u[:, 0].cpu().detach().numpy().reshape(-1, 1))

        plt.title(filename2)
        plt.savefig(filename2)
        plt.show()

    # 文章摘要中提到，最小化负对数似然估计来不断更新自适应权重
    def likelihood_loss(self, loss_e, loss_l):
        loss = torch.exp(-self.x_f_s) * loss_e.detach() + self.x_f_s \
               + torch.exp(-self.x_label_s) * loss_l.detach() + self.x_label_s
        return loss

    def true_loss(self, loss_e, loss_l):
        return torch.exp(-self.x_f_s.detach()) * loss_e + torch.exp(-self.x_label_s.detach()) * loss_l

    # compute backward loss
    def epoch_loss(self):
        x_f_A1 = torch.cat((self.x_f_N_A1, self.x_f_M_A1), dim=0)
        x_f_A2 = torch.cat((self.x_f_N_A2, self.x_f_M_A2), dim=0)
        #-----------------
        # # 区域内物理值预测，数量级是一样的吗
        # loss_equation_A1 = torch.mean(self.x_f_loss_fun_A1(x_f_A1, self.train_U) ** 2)
        # loss_equation_A2 = torch.mean(self.x_f_loss_fun_A2(x_f_A2, self.train_U) ** 2)
        # # 区域内的损失函数
        # loss_equation = loss_equation_A1 + loss_equation_A2
        # # 边界物理值预测与标签的均方差，需要修改,考虑到x_f_loss_fun_Ax中具有导数项，故将边界的损失函数写入到区域内计算
        # loss_val_b1 = torch.mean(self.loss_b1(self.b1, self.train_U)** 2)#dAb1/dy
        # loss_val_b4 = torch.mean(self.loss_b4(self.b4, self.train_U) ** 2)
        # loss_val_b57 = 10*torch.mean(self.loss_b57(self.b5, self.b7,self.train_U) ** 2)
        # loss_val_b68 = 10*torch.mean(self.loss_b68(self.b6,self.b8, self.train_U) ** 2)
        # # 边界上的损失函数
        # loss_label =10*(loss_val_b1+loss_val_b4+loss_val_b57+loss_val_b68)
        #----------------

        # ----------------
        # 区域内物理值预测，数量级是一样的吗
        loss_equation_A1 = (self.x_f_loss_fun_A1(x_f_A1, self.train_U).sum())
        loss_equation_A2 = (self.x_f_loss_fun_A2(x_f_A2, self.train_U).sum())
        # 区域内的损失函数
        loss_equation = loss_equation_A1 + loss_equation_A2
        # 边界物理值预测与标签的均方差，需要修改,考虑到x_f_loss_fun_Ax中具有导数项，故将边界的损失函数写入到区域内计算
        loss_val_b1 = (self.loss_b1(self.b1, self.train_U)) # dAb1/dy
        loss_val_b2 = (self.loss_b2(self.b2, self.train_U))
        loss_val_b3 = (self.loss_b3(self.b3, self.train_U))
        loss_val_b4 = (self.loss_b4(self.b4, self.train_U))
        loss_val_b57 = (self.loss_b57(self.b5, self.b7, self.train_U))
        loss_val_b68 = (self.loss_b68(self.b6, self.b8, self.train_U))
        # 边界上的损失函数
        loss_label = (loss_val_b1+loss_val_b2+loss_val_b3 + loss_val_b4+loss_val_b57+loss_val_b68 )
        # ----------------

        # 不执行
        if self.start_loss_collect:
            self.x_f_loss_collect.append([self.net.iter, loss_equation.item()])
            self.x_label_loss_collect.append([self.net.iter, loss_label.item()])
        return loss_equation, loss_label

    # compute loss in sensors way
    def custom_train(self, method):
        alpha_ann = 0.5
        method = method
        lambd = 1
        lambds = [];

        x_f_A1 = torch.cat((self.x_f_N_A1, self.x_f_M_A1), dim=0)
        x_f_A2 = torch.cat((self.x_f_N_A2, self.x_f_M_A2), dim=0)
        X_train = torch.cat((x_f_A1, x_f_A2), dim=0)

        #这里的残差采用了平方后取均值，原始的实验表明可能平方后求和的loss更有效
        l_reg_A1 = torch.mean(self.x_f_loss_fun_A1(x_f_A1, self.net))
        l_reg_A2 = torch.mean(self.x_f_loss_fun_A2(x_f_A2, self.net))
        l_reg = l_reg_A1 + l_reg_A2
        # 这里的残差采用了平方后取均值，原始的实验表明可能平方后求和的loss更有效,后续实验可以尝试改回平方后sum
        loss_val_b1 = (self.loss_b1(self.b1, self.net))  # dAb1/dy
        loss_val_b2 = (self.loss_b2(self.b2, self.net))
        loss_val_b3 = (self.loss_b3(self.b3, self.net))
        loss_val_b4 = (self.loss_b4(self.b4, self.net))
        loss_val_b57 = (self.loss_b57(self.b5, self.b7, self.net))
        loss_val_b68 = (self.loss_b68(self.b6, self.b8, self.net))
        l_bc = (loss_val_b1+loss_val_b2+loss_val_b3 + loss_val_b4+loss_val_b57+loss_val_b68 )

        with torch.no_grad():
            # stdr, kurtr = loss_grad_stats(l_reg, self.net)
            # stdb, kurtb = loss_grad_stats(l_bc, self.net)
            # maxr, meanr = loss_grad_max_mean(l_reg, self.net)
            # maxb, meanb = loss_grad_max_mean(l_bc, self.net, lambg=lambd)
            if method == 2:
                # inverse dirichlet
                stdr, kurtr = loss_grad_stats(l_reg, self.net)
                stdb, kurtb = loss_grad_stats(l_bc, self.net)
                lamb_hat = stdr / stdb
                lambd = (1 - alpha_ann) * lambd + alpha_ann * lamb_hat
            elif method == 1:
                # max/avg
                maxr, meanr = loss_grad_max_mean(l_reg, self.net)
                maxb, meanb = loss_grad_max_mean(l_bc, self.net, lambg=lambd)
                lamb_hat = maxr / meanb
                lambd = (1 - alpha_ann) * lambd + alpha_ann * lamb_hat
            elif method == 3:
                # mean + std weighing
                stdr, kurtr = loss_grad_stats(l_reg, self.net)
                stdb, kurtb = loss_grad_stats(l_bc, self.net)
                maxr, meanr = loss_grad_max_mean(l_reg, self.net)
                maxb, meanb = loss_grad_max_mean(l_bc, self.net, lambg=lambd)
                covr = stdr + maxr
                covb = stdb + meanb
                lamb_hat = covr / covb
                lambd = (1 - alpha_ann) * lambd + alpha_ann * lamb_hat
            elif method == 5:
                # kurtosis based weighing
                stdr, kurtr = loss_grad_stats(l_reg, self.net)
                stdb, kurtb = loss_grad_stats(l_bc, self.net)
                covr = stdr / kurtr
                covb = stdb / kurtb
                lamb_hat = covr / covb
                lambd = (1 - alpha_ann) * lambd + alpha_ann * lamb_hat
            elif method == 4:
                # mean * std weighing
                stdr, kurtr = loss_grad_stats(l_reg, self.net)
                stdb, kurtb = loss_grad_stats(l_bc, self.net)
                maxr, meanr = loss_grad_max_mean(l_reg, self.net)
                maxb, meanb = loss_grad_max_mean(l_bc, self.net, lambg=lambd)
                covr = stdr * meanr
                covb = stdb * meanb
                lamb_hat = covr / covb
                lambd = (1 - alpha_ann) * lambd + alpha_ann * lamb_hat
            else:
                # uniform weighting
                lambd = 1;

        if (method == 0):
            return l_reg, l_bc
        elif (method == 1 or method == 2 or method == 3 or method == 4 or method == 5):
            return l_reg, lambd * l_bc

    # computer backward loss
    def LBGFS_epoch_loss(self):
        self.optimizer_LBGFS.zero_grad()
        x_f_A1 = torch.cat((self.x_f_N_A1, self.x_f_M_A1), dim=0)
        x_f_A2 = torch.cat((self.x_f_N_A2, self.x_f_M_A2), dim=0)

        # ----------------
        # loss_equation_A1 = torch.mean(self.x_f_loss_fun_A1(x_f_A1, self.train_U) ** 2)
        # loss_equation_A2 = torch.mean(self.x_f_loss_fun_A2(x_f_A2, self.train_U) ** 2)
        # loss_equation = loss_equation_A1 + loss_equation_A2
        #
        # # 边界物理值预测与标签的均方差，需要修改,考虑到x_f_loss_fun_Ax中具有导数项，故将边界的损失函数写入到区域内计算
        # loss_val_b1 = torch.mean(self.loss_b1(self.b1, self.train_U) ** 2)
        # loss_val_b4 = torch.mean(self.loss_b4(self.b4, self.train_U) ** 2)
        # loss_val_b57 = 10*torch.mean(self.loss_b57(self.b5, self.b7, self.train_U) ** 2)
        # loss_val_b68 = 10*torch.mean(self.loss_b68(self.b6, self.b8, self.train_U) ** 2)
        # # 边界上的损失函数
        # loss_label = 100*(loss_val_b1 + loss_val_b4+loss_val_b57+loss_val_b68)
        # ----------------
        loss_equation_A1 = (self.x_f_loss_fun_A1(x_f_A1, self.train_U)).sum()
        loss_equation_A2 = (self.x_f_loss_fun_A2(x_f_A2, self.train_U)).sum()
        loss_equation = loss_equation_A1 + loss_equation_A2

        # 边界物理值预测与标签的均方差，需要修改,考虑到x_f_loss_fun_Ax中具有导数项，故将边界的损失函数写入到区域内计算
        loss_val_b1 = (self.loss_b1(self.b1, self.train_U))
        loss_val_b2 = (self.loss_b2(self.b2, self.train_U))
        loss_val_b3 = (self.loss_b3(self.b3, self.train_U))
        loss_val_b4 = (self.loss_b4(self.b4, self.train_U))
        loss_val_b57 = (self.loss_b57(self.b5, self.b7, self.train_U))
        loss_val_b68 = (self.loss_b68(self.b6, self.b8, self.train_U))
        # 边界上的损失函数
        loss_label =  (loss_val_b1+loss_val_b2+loss_val_b3 + loss_val_b4 + loss_val_b57 + loss_val_b68)

        if self.start_loss_collect:
            self.x_f_loss_collect.append([self.net.iter, loss_equation.item()])
            self.x_label_loss_collect.append([self.net.iter, loss_label.item()])

        loss = self.true_loss(loss_equation, loss_label)
        #loss = loss_equation + loss_label
        loss.backward(retain_graph=True)
        self.net.iter += 1
        #print('Iter:', self.net.iter, 'Loss:', loss.item())
        return loss

    def evaluate(self):
        # self.x_test = is_cuda(self.x_test)
        # pred = self.train_U(self.x_test).cpu().detach().numpy()
        # exact = torch.cat((self.x_test_exact_Adx,self.x_test_exact_Ady), axis = 1).cpu().detach().numpy()
        self.x_cooper_test = is_cuda(self.x_test_copper)
        pred_cop = self.train_U(self.x_cooper_test).cpu().detach().numpy()
        exact_cop = torch.cat((self.x_test_copper_Adx, self.x_test_copper_Ady), axis=1).cpu().detach().numpy()
        error = np.linalg.norm(pred_cop - exact_cop, 2) / np.linalg.norm(exact_cop, 2)
        return error

    #原始PINN，无移动点、无自适应loss权重
    def run_baseline(self):
        self.x_f_s = nn.Parameter(self.x_f_s, requires_grad=True)
        self.x_label_s = nn.Parameter(self.x_label_s, requires_grad=True)

        for move_count in range(AM_count):
            self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                     max_iter=lbgfs_iter)
            optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)

            pbar = trange(adam_iter, ncols=100)
            for i in pbar:
                self.s_collect.append([self.net.iter, self.x_f_s.item(), self.x_label_s.item()])
                loss_e, loss_label = self.epoch_loss()
                optimizer_adam.zero_grad()
                loss = self.true_loss(loss_e, loss_label)
                loss.backward(retain_graph=True)
                optimizer_adam.step()

                self.net.iter += 1
                pbar.set_postfix({'Iter': self.net.iter,
                                  'Loss-sum': '{0:.2e}'.format(loss.item())

                                  })

            print('Adam done!')
            error = self.evaluate()
            print('change_counts', move_count, 'Test_L2error:', '{0:.2e}'.format(error), 'Loss-pde', '{0:.2e}'.format(loss_e.item()),
                                  'Loss-label', '{0:.2e}'.format(loss_label.item()),)
            self.x_test_estimate_collect.append([move_count, '{0:.2e}'.format(error)])
            self.render_res(move_count, pic_name)
            self.render_residual(move_count, pic_name)

    #AW-PINN，无移动点、有自适应loss权重
    def run_AW(self):
        self.x_f_s = nn.Parameter(self.x_f_s, requires_grad=True)
        self.x_label_s = nn.Parameter(self.x_label_s, requires_grad=True)

        for move_count in range(AM_count):
            self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                     max_iter=lbgfs_iter)

            optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
            optimizer_adam_weight = torch.optim.Adam([self.x_f_s] + [self.x_label_s],
                                                     lr=AW_lr)

            pbar = trange(adam_iter, ncols=100)
            for i in pbar:
                self.s_collect.append([self.net.iter, self.x_f_s.item(), self.x_label_s.item()])
                loss_e, loss_label = self.epoch_loss()
                optimizer_adam.zero_grad()
                loss = self.true_loss(loss_e, loss_label)
                loss.backward(retain_graph=True)
                optimizer_adam.step()
                self.net.iter += 1
                pbar.set_postfix({'Iter': self.net.iter,
                                  'Loss': '{0:.2e}'.format(loss.item())
                                  })
                optimizer_adam_weight.zero_grad()
                loss = self.likelihood_loss(loss_e, loss_label)
                loss.backward()
                optimizer_adam_weight.step()

            print('Adam done!')
            error = self.evaluate()
            print('change_counts', move_count, 'Test_L2error:', '{0:.2e}'.format(error))
            self.x_test_estimate_collect.append([move_count, '{0:.2e}'.format(error)])
            self.render_res(move_count, pic_name)

    #R/GAM-PINN，有残差or梯度移动点、无自适应loss权重
    def run_AM(self):
        self.x_f_s = nn.Parameter(self.x_f_s, requires_grad=True)
        self.x_label_s = nn.Parameter(self.x_label_s, requires_grad=True)

        for move_count in range(AM_count):
            self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                     max_iter=lbgfs_iter)

            optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
            optimizer_adam_weight = torch.optim.Adam([self.x_f_s] + [self.x_label_s],
                                                     lr=AW_lr)

            pbar = trange(adam_iter, ncols=100)
            for i in pbar:
                self.s_collect.append([self.net.iter, self.x_f_s.item(), self.x_label_s.item()])
                # inner loss and boundary lpss
                loss_e, loss_label = self.epoch_loss()

                optimizer_adam.zero_grad()
                loss = self.true_loss(loss_e, loss_label)
                loss.backward(retain_graph=True)
                optimizer_adam.step()

                self.net.iter += 1
                pbar.set_postfix({'Iter': self.net.iter,
                                  'Loss': '{0:.2e}'.format(loss.item())
                                  })

            print('Adam done!')
            error = self.evaluate()
            print('change_counts', move_count, 'Test_L2error:', '{0:.2e}'.format(error))
            self.x_test_estimate_collect.append([move_count, '{0:.2e}'.format(error)])


            # 执行该部分代码
            if AM_type == 0:
                x_init_A1 = random_fun(2000, lb_A1, ub_A1)
                x_init_A2 = random_fun(2000, lb_A2, ub_A2)
                x_init_residual_A1 = abs(self.x_f_loss_fun_A1(x_init_A1, self.train_U))
                x_init_residual_A2 = abs(self.x_f_loss_fun_A2(x_init_A2, self.train_U))
                x_init_residual_A1 = x_init_residual_A1.cpu().detach().numpy()
                x_init_residual_A2 = x_init_residual_A2.cpu().detach().numpy()

                err_eq_A1 = np.power(x_init_residual_A1, AM_K) / np.power(x_init_residual_A1, AM_K).mean()
                err_eq_A2 = np.power(x_init_residual_A2, AM_K) / np.power(x_init_residual_A2, AM_K).mean()

                err_eq_A1 = err_eq_A1.reshape(x_init_residual_A1.size,1)
                err_eq_A2 = err_eq_A2.reshape(x_init_residual_A2.size,1)


                err_eq_normalized_A1 = (err_eq_A1 / sum(err_eq_A1))[:, 0]
                err_eq_normalized_A2 = (err_eq_A2 / sum(err_eq_A2))[:, 0]
                #按照概率选取
                if not np.isnan(err_eq_normalized_A1).any():
                    X_ids_A1 = np.random.choice(a=len(x_init_A1), size=M, replace=False, p=err_eq_normalized_A1)
                    self.x_f_M_A1 = x_init_A1[X_ids_A1]  # (1000,2)

                if not np.isnan(err_eq_normalized_A2).any():
                    X_ids_A2 = np.random.choice(a=len(x_init_A2), size=M, replace=False, p=err_eq_normalized_A2)
                    self.x_f_M_A2 = x_init_A2[X_ids_A2]

            elif AM_type == 1:
                x_init_A1 = random_fun(2000, lb_A1, ub_A1)
                x_init_A2 = random_fun(2000, lb_A2, ub_A2)
                x_init_A1 = Variable(x_init_A1, requires_grad=True)
                x_init_A2 = Variable(x_init_A2, requires_grad=True)
                u_A1 = self.train_U(x_init_A1)
                A1dx = torch.autograd.grad(u_A1[:, [0]], x_init_A1, grad_outputs=torch.ones_like(u_A1[:, [0]]),
                                           create_graph=True)[0]
                grad_A1dxdx = A1dx[:, [0]].squeeze()
                grad_A1dxdy = A1dx[:, [1]].squeeze()
                A1dx = torch.sqrt(1 + grad_A1dxdx ** 2 + grad_A1dxdy ** 2).cpu().detach().numpy()
                err_A1dx = np.power(A1dx, AM_K) / np.power(A1dx, AM_K).mean()
                p_gradA1 = (err_A1dx / sum(err_A1dx))
                X_A1ids = np.random.choice(a=len(x_init_A1), size=M, replace=False, p=p_gradA1)
                self.x_f_M_A1 = x_init_A1[X_A1ids]

                u_A2 = self.train_U(x_init_A2)
                A2dx = torch.autograd.grad(u_A2[:, [0]], x_init_A2, grad_outputs=torch.ones_like(u_A2[:, [0]]),
                                           create_graph=True)[0]
                grad_A2dxdx = A2dx[:, [0]].squeeze()
                grad_A2dxdy = A2dx[:, [1]].squeeze()
                A2dx = torch.sqrt(1 + grad_A2dxdx ** 2 + grad_A2dxdy ** 2).cpu().detach().numpy()
                err_A2dx = np.power(A2dx, AM_K) / np.power(A2dx, AM_K).mean()
                p_gradA2 = (err_A2dx / sum(err_A2dx))
                X_A2ids = np.random.choice(a=len(x_init_A2), size=M, replace=False, p=p_gradA2)
                self.x_f_M_A2 = x_init_A2[X_A2ids]

            elif AM_type == 2:
                print('One loop done!')

            self.render_res(move_count, pic_name)

    #R/GAM-AW-PINN，有残差or梯度移动点、有自适应loss权重
    def run_AM_AW(self):
        self.x_f_s = nn.Parameter(self.x_f_s, requires_grad=True)
        self.x_label_s = nn.Parameter(self.x_label_s, requires_grad=True)

        for move_count in range(AM_count):
            self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                     max_iter=lbgfs_iter)

            optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
            optimizer_adam_weight = torch.optim.Adam([self.x_f_s] + [self.x_label_s],
                                                     lr=AW_lr)

            pbar = trange(adam_iter, ncols=100)
            for i in pbar:
                self.s_collect.append([self.net.iter, self.x_f_s.item(), self.x_label_s.item()])
                # inner loss and boundary lpss
                loss_e, loss_label = self.epoch_loss()
                optimizer_adam.zero_grad()
                loss = self.true_loss(loss_e, loss_label)
                #loss = loss_e+label_weight*loss_label
                loss.backward(retain_graph=True)
                optimizer_adam.step()

                self.net.iter += 1
                pbar.set_postfix({'Iter': self.net.iter,
                                  'Loss': '{0:.2e}'.format(loss.item())
                                  })

                optimizer_adam_weight.zero_grad()
                loss = self.likelihood_loss(loss_e, loss_label)
                loss.backward()
                optimizer_adam_weight.step()

            print('Adam done!')
            error = self.evaluate()
            print('change_counts', move_count, 'Test_L2error:', '{0:.2e}'.format(error))
            self.x_test_estimate_collect.append([move_count, '{0:.2e}'.format(error)])


            # 执行该部分代码
            if AM_type == 0:
                x_init_A1 = random_fun(2000, lb_A1, ub_A1)
                x_init_A2 = random_fun(2000, lb_A2, ub_A2)
                x_init_residual_A1 = abs(self.x_f_loss_fun_A1(x_init_A1, self.train_U))
                x_init_residual_A2 = abs(self.x_f_loss_fun_A2(x_init_A2, self.train_U))
                x_init_residual_A1 = x_init_residual_A1.cpu().detach().numpy()
                x_init_residual_A2 = x_init_residual_A2.cpu().detach().numpy()

                err_eq_A1 = np.power(x_init_residual_A1, AM_K) / np.power(x_init_residual_A1, AM_K).mean()
                err_eq_A2 = np.power(x_init_residual_A2, AM_K) / np.power(x_init_residual_A2, AM_K).mean()

                err_eq_A1 = err_eq_A1.reshape(x_init_residual_A1.size,1)
                err_eq_A2 = err_eq_A2.reshape(x_init_residual_A2.size,1)

                err_eq_normalized_A1 = (err_eq_A1 / sum(err_eq_A1))[:, 0]
                err_eq_normalized_A2 = (err_eq_A2 / sum(err_eq_A2))[:, 0]
                #按照概率选取
                if not np.isnan(err_eq_normalized_A1).any():
                    X_ids_A1 = np.random.choice(a=len(x_init_A1), size=M, replace=False, p=err_eq_normalized_A1)
                    self.x_f_M_A1 = x_init_A1[X_ids_A1]  # (1000,2)

                if not np.isnan(err_eq_normalized_A2).any():
                    X_ids_A2 = np.random.choice(a=len(x_init_A2), size=M, replace=False, p=err_eq_normalized_A2)
                    self.x_f_M_A2 = x_init_A2[X_ids_A2]

            elif AM_type == 1:
                x_init_A1 = random_fun(2000, lb_A1, ub_A1)
                x_init_A2 = random_fun(2000, lb_A2, ub_A2)
                x_init_A1 = Variable(x_init_A1, requires_grad=True)
                x_init_A2 = Variable(x_init_A2, requires_grad=True)
                u_A1 = self.train_U(x_init_A1)
                A1dx = torch.autograd.grad(u_A1[:, [0]], x_init_A1, grad_outputs=torch.ones_like(u_A1[:, [0]]),
                                           create_graph=True)[0]
                grad_A1dxdx = A1dx[:, [0]].squeeze()
                grad_A1dxdy = A1dx[:, [1]].squeeze()
                A1dx = torch.sqrt(1 + grad_A1dxdx ** 2 + grad_A1dxdy ** 2).cpu().detach().numpy()
                err_A1dx = np.power(A1dx, AM_K) / np.power(A1dx, AM_K).mean()
                p_gradA1 = (err_A1dx / sum(err_A1dx))
                X_A1ids = np.random.choice(a=len(x_init_A1), size=M, replace=False, p=p_gradA1)
                self.x_f_M_A1 = x_init_A1[X_A1ids]

                u_A2 = self.train_U(x_init_A2)
                A2dx = torch.autograd.grad(u_A2[:, [0]], x_init_A2, grad_outputs=torch.ones_like(u_A2[:, [0]]),
                                           create_graph=True)[0]
                grad_A2dxdx = A2dx[:, [0]].squeeze()
                grad_A2dxdy = A2dx[:, [1]].squeeze()
                A2dx = torch.sqrt(1 + grad_A2dxdx ** 2 + grad_A2dxdy ** 2).cpu().detach().numpy()
                err_A2dx = np.power(A2dx, AM_K) / np.power(A2dx, AM_K).mean()
                p_gradA2 = (err_A2dx / sum(err_A2dx))
                X_A2ids = np.random.choice(a=len(x_init_A2), size=M, replace=False, p=p_gradA2)
                self.x_f_M_A2 = x_init_A2[X_A2ids]

            elif AM_type == 2:
                print('One loop done!')

            self.render_res(move_count, pic_name)

    def run_AM_sensors(self):
        self.x_f_s = nn.Parameter(self.x_f_s, requires_grad=True)
        self.x_label_s = nn.Parameter(self.x_label_s, requires_grad=True)

        for move_count in range(AM_count):
            self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                     max_iter=lbgfs_iter)

            optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
            optimizer_adam_weight = torch.optim.Adam([self.x_f_s] + [self.x_label_s],
                                                     lr=AW_lr)

            pbar = trange(adam_iter, ncols=100)
            for i in pbar:
                self.s_collect.append([self.net.iter, self.x_f_s.item(), self.x_label_s.item()])
                # inner loss and boundary lpss
                loss_e, loss_label = self.custom_train(self.method)
                optimizer_adam.zero_grad()
                loss = self.true_loss(loss_e, loss_label)
                loss.backward(retain_graph=True)
                optimizer_adam.step()

                self.net.iter += 1
                pbar.set_postfix({'Iter': self.net.iter,
                                  'Loss': '{0:.2e}'.format(loss.item())
                                  })

            print('Adam done!')
            error = self.evaluate()
            print('change_counts', move_count, 'Test_L2error:', '{0:.2e}'.format(error))
            self.x_test_estimate_collect.append([move_count, '{0:.2e}'.format(error)])

            # 执行该部分代码
            if AM_type == 0:
                x_init_A1 = random_fun(2000, lb_A1, ub_A1)
                x_init_A2 = random_fun(2000, lb_A2, ub_A2)
                x_init_residual_A1 = abs(self.x_f_loss_fun_A1(x_init_A1, self.train_U))
                x_init_residual_A2 = abs(self.x_f_loss_fun_A2(x_init_A2, self.train_U))
                x_init_residual_A1 = x_init_residual_A1.cpu().detach().numpy()
                x_init_residual_A2 = x_init_residual_A2.cpu().detach().numpy()

                err_eq_A1 = np.power(x_init_residual_A1, AM_K) / np.power(x_init_residual_A1, AM_K).mean()
                err_eq_A2 = np.power(x_init_residual_A2, AM_K) / np.power(x_init_residual_A2, AM_K).mean()

                err_eq_A1 = err_eq_A1.reshape(x_init_residual_A1.size, 1)
                err_eq_A2 = err_eq_A2.reshape(x_init_residual_A2.size, 1)

                err_eq_normalized_A1 = (err_eq_A1 / sum(err_eq_A1))[:, 0]
                err_eq_normalized_A2 = (err_eq_A2 / sum(err_eq_A2))[:, 0]
                # 按照概率选取
                if not np.isnan(err_eq_normalized_A1).any():
                    X_ids_A1 = np.random.choice(a=len(x_init_A1), size=M, replace=False, p=err_eq_normalized_A1)
                    self.x_f_M_A1 = x_init_A1[X_ids_A1]  # (1000,2)

                if not np.isnan(err_eq_normalized_A2).any():
                    X_ids_A2 = np.random.choice(a=len(x_init_A2), size=M, replace=False, p=err_eq_normalized_A2)
                    self.x_f_M_A2 = x_init_A2[X_ids_A2]

            elif AM_type == 1:
                x_init_A1 = random_fun(2000, lb_A1, ub_A1)
                x_init_A2 = random_fun(2000, lb_A2, ub_A2)
                x_init_A1 = Variable(x_init_A1, requires_grad=True)
                x_init_A2 = Variable(x_init_A2, requires_grad=True)
                u_A1 = self.train_U(x_init_A1)
                A1dx = torch.autograd.grad(u_A1[:, [0]], x_init_A1, grad_outputs=torch.ones_like(u_A1[:, [0]]),
                                           create_graph=True)[0]
                grad_A1dxdx = A1dx[:, [0]].squeeze()
                grad_A1dxdy = A1dx[:, [1]].squeeze()
                A1dx = torch.sqrt(1 + grad_A1dxdx ** 2 + grad_A1dxdy ** 2).cpu().detach().numpy()
                err_A1dx = np.power(A1dx, AM_K) / np.power(A1dx, AM_K).mean()
                p_gradA1 = (err_A1dx / sum(err_A1dx))
                X_A1ids = np.random.choice(a=len(x_init_A1), size=M, replace=False, p=p_gradA1)
                self.x_f_M_A1 = x_init_A1[X_A1ids]

                u_A2 = self.train_U(x_init_A2)
                A2dx = torch.autograd.grad(u_A2[:, [0]], x_init_A2, grad_outputs=torch.ones_like(u_A2[:, [0]]),
                                           create_graph=True)[0]
                grad_A2dxdx = A2dx[:, [0]].squeeze()
                grad_A2dxdy = A2dx[:, [1]].squeeze()
                A2dx = torch.sqrt(1 + grad_A2dxdx ** 2 + grad_A2dxdy ** 2).cpu().detach().numpy()
                err_A2dx = np.power(A2dx, AM_K) / np.power(A2dx, AM_K).mean()
                p_gradA2 = (err_A2dx / sum(err_A2dx))
                X_A2ids = np.random.choice(a=len(x_init_A2), size=M, replace=False, p=p_gradA2)
                self.x_f_M_A2 = x_init_A2[X_A2ids]

            elif AM_type == 2:
                print('No movable point')

            self.render_res(move_count, pic_name)

    def train(self):
        self.x_f_s = is_cuda(torch.tensor(0.).float())
        self.x_label_s = is_cuda(torch.tensor(0.).float())

        start_time = time.time()
        if model_type == 0:
            self.run_baseline()
        elif model_type == 1:
            self.run_AW()
        elif model_type == 2:
            self.run_AM()
        elif model_type == 3:
            self.run_AM_AW()
        elif model_type == 4:
            self.run_AM_sensors()

        elapsed = time.time() - start_time
        print('Training time: %.2f' % elapsed)

    def save(self, dirname):
        # 创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
        timestr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        modelfname = os.path.join(dirname, model_name + timestr)
        # torch.save(self.nn.state_dict(), modelfname)
        torch.save(self.net, modelfname)
        # rfile = os.path.join(dirname, 'results.npz' +timestr)
        # x, y, u = self.get_plot_data()
        # x, y, u, pn_copper, x_copper, y_copper = self.get_plot_data()
        # np.savez(rfile, x=x, y=y, u=u)

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

def tensor(x, **kw):
    '''Returns a suitable device specific tensor.
    '''
    return torch.tensor(x, dtype=torch.float32,  **kw)

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

if __name__ == '__main__':
    # PMD param
    Pi = np.pi
    # 真空磁导率
    u0 = tensor(4 * Pi * 1e-7, requires_grad=True)
    br = 1.25
    ur = br*u0
    sigma = 57000000
    R1 = 0.03   # Inner radius of magnets
    R2 = 0.06   # Outer radius of magnets
    R3 = 0.015  # Inner radius of conducting plate
    R4 = 0.075  # Outer radius of conduction plate
    L = R2-R1
    Rm = (R1+R2)/2
    slip = 150
    omega = slip * 2 * Pi / 60
    p = 5  # ploe-pair
    alpha = 1  # pole-arc to pole-pitch ratio
    beta = p / Rm   # 极对数除以永磁体平均半径，Lubin论文公式（2）
    thickness_mag = 0.01
    thickness_air = 0.003
    thickness_copper = 0.005
    length_left = 0
    # 只求解一个磁极距离的
    length_right = alpha * Rm * Pi / p
    height_bottom = 0
    height_top = thickness_mag + thickness_air + thickness_copper
    fitness = 0
    b = thickness_mag
    c = thickness_air
    d = thickness_copper
    field_res = 20     #

    # exact_solution
    # lubin论文
    K = ((4 * br * Rm) / (Pi * p)) * (torch.sin(tensor(alpha * Pi / 2, requires_grad=True)))
    # exact_A1 = lambda x, y: K * (1 - (( torch.sinh(tensor(beta * (c + d))) * torch.cosh(tensor(beta * y).unsqueeze(-1))) / torch.sinh(
    #     tensor(beta * (b + c + d))))) * torch.cos(tensor(beta * x).unsqueeze(-1))
    # exact_A2 = lambda x, y: K * (torch.sinh(tensor(beta * b)) * torch.cosh(tensor(beta * (y - (b + c + d))).unsqueeze(-1)) * torch.cos(
    #     tensor(beta * x).unsqueeze(-1))) / torch.sinh(tensor(beta * (b + c + d)))

    lb_A1 = np.array([length_left+fitness, height_bottom+fitness])
    ub_A1 = np.array([length_right-fitness, thickness_mag-fitness])
    lb_A2 = np.array([length_left+fitness, thickness_mag+fitness])
    ub_A2 = np.array([length_right-fitness, height_top-fitness])
    # 铜盘区域的坐标
    lb_copper = np.array([length_left, thickness_mag+thickness_air])
    ub_copper = np.array([length_right, thickness_mag + thickness_air + thickness_copper])


    # net网络参数定义
    neurons = 50
    layers = [2, neurons, neurons, neurons, neurons, neurons, 2]
    method = 1
    N = 500     # 不移动点
    M = 1000    # 移动点
    Nbc = 200    #一条边界上的点
    # 不同优化器迭代次数与学习率的设置
    adam_iter, lbgfs_iter = 100, 10000
    adam_lr, lbgfs_lr = 0.001, 0.5
    # 设置模型权重调节类型，
    model_type = 0  # 0:baseline  1:AW  2:AM  3:AM_AW  4:AM_sensors
    # 采样方式设置
    AM_type = 2  # 0:RAM  1:WAM  #2:None
    AM_K = 1
    AM_count = 20
    AW_lr = 0.001
    pic_name = "tim_PMC-PINN"
    model_name = "model_tim_PMC-PINN.pt"

    # test data测试数据的定义域
    # A1 test data
    x1 = np.expand_dims(np.linspace(length_left, length_right, 128), axis=1)
    x2 = np.expand_dims(np.linspace(height_bottom, thickness_mag, 128), axis=1)
    X1, X2 = np.meshgrid(x1, x2)
    x_test_np_A1 = np.concatenate((np.vstack(np.expand_dims(X1, axis=2)), np.vstack(np.expand_dims(X2, axis=2))), axis=-1) #16384,2
    # A2 test data
    x1 = np.expand_dims(np.linspace(length_left, length_right, 128), axis=1)
    x2 = np.expand_dims(np.linspace(thickness_mag, height_top, 128), axis=1)
    X1, X2 = np.meshgrid(x1, x2)
    x_test_np_A2 = np.concatenate((np.vstack(np.expand_dims(X1, axis=2)), np.vstack(np.expand_dims(X2, axis=2))), axis=-1)
    # 通过解析计算式，计算的结果
    input_A1_x = tensor(x_test_np_A1[:, [0]], requires_grad=True)
    input_A1_y = tensor(x_test_np_A1[:, [1]], requires_grad=True)
    input_A2_x = tensor(x_test_np_A2[:, [0]], requires_grad=True)
    input_A2_y = tensor(x_test_np_A2[:, [1]], requires_grad=True)

    solution_A1 = exact_A1(input_A1_x, input_A1_y)
    solution_A2 = exact_A2(input_A2_x, input_A2_y)
    solution_A1dx = grad(solution_A1, input_A1_x, grad_outputs=torch.ones_like(solution_A1), create_graph=True)
    solution_A1dy = grad(solution_A1, input_A1_y, grad_outputs=torch.ones_like(solution_A1), create_graph=True)
    solution_A2dx = grad(solution_A2, input_A2_x, grad_outputs=torch.ones_like(solution_A2), create_graph=True)
    solution_A2dy = grad(solution_A2, input_A2_y, grad_outputs=torch.ones_like(solution_A2), create_graph=True)
    # copper test data
    x1 = np.expand_dims(np.linspace(length_left, length_right, 128), axis=1)
    x2 = np.expand_dims(np.linspace(thickness_mag + thickness_air, height_top, 128), axis=1)
    X1, X2 = np.meshgrid(x1, x2)
    x_test_copper_A2 = np.concatenate((np.vstack(np.expand_dims(X1, axis=2)), np.vstack(np.expand_dims(X2, axis=2))),
                                      axis=-1)
    # copper cordinate
    input_copper_A2_x = tensor(x_test_copper_A2[:, [0]], requires_grad=True)
    input_copper_A2_y = tensor(x_test_copper_A2[:, [1]], requires_grad=True)
    # copper cordinate
    x_test_copper_A2 = torch.cat((input_copper_A2_x, input_copper_A2_y), axis=1)
    # copper solution
    solution_copper_A2 = exact_A2(input_copper_A2_x, input_copper_A2_y)
    solution_copper_A2dx = \
    grad(solution_copper_A2, input_copper_A2_x, grad_outputs=torch.ones_like(solution_copper_A2), create_graph=True)[0]
    solution_copper_A2dy = \
    grad(solution_copper_A2, input_copper_A2_y, grad_outputs=torch.ones_like(solution_copper_A2), create_graph=True)[0]
    # x_test_np = np.concatenate((x_test_np_A1, x_test_np_A2), axis=0)#32768,2
    # solution = np.concatenate((solution_A1, solution_A2), axis=0)

    # x_test = is_cuda(torch.from_numpy(x_test_np).float())
    # x_test_exact = is_cuda(torch.from_numpy(solution).float())
    x_test_A1 = torch.cat((input_A1_x, input_A1_y), axis=1)
    x_test_A2 = torch.cat((input_A2_x, input_A2_y), axis=1)

    x_test = torch.cat((x_test_A1, x_test_A2), axis=0)
    x_test_exact_Adx = torch.cat((solution_A1dx[0], solution_A2dx[0]), axis=0)
    x_test_exact_Ady = torch.cat((solution_A1dy[0], solution_A2dy[0]), axis=0)

    # bc data边界上的点
    # A1左侧
    x1_boundary_left_A1 = torch.cat((torch.full([Nbc, 1], length_left, device = device), torch.full([Nbc, 1], height_bottom, device = device) + torch.rand([Nbc, 1], device = device) * thickness_mag), dim=1) #leftA1
    # A2左侧
    x1_boundary_left_A2 = torch.cat((torch.full([Nbc, 1], length_left, device = device), torch.full([Nbc, 1], thickness_mag, device = device) + torch.rand([Nbc, 1], device = device) * (thickness_air+thickness_copper)), dim=1) #leftA2
    # A1右侧
    x1_boundary_right_A1 = torch.cat((torch.full([Nbc, 1], length_right, device = device), torch.full([Nbc, 1], height_bottom, device = device) + torch.rand([Nbc, 1], device = device) * thickness_mag), dim=1) #rightA1
    # A2右侧
    x1_boundary_right_A2 = torch.cat((torch.full([Nbc, 1], length_right, device = device), torch.full([Nbc, 1], thickness_mag, device = device) + torch.rand([Nbc, 1], device = device) * (thickness_air+thickness_copper)), dim=1) #rightA2
    # A1下侧
    x2_boundary_left = torch.cat((torch.full([Nbc, 1], length_left, device = device) + torch.rand([Nbc, 1], device = device) * (length_right-length_left), torch.full([Nbc, 1], height_bottom, device = device)), dim=1) #bottom
    # A2上侧
    x2_boundary_right = torch.cat((torch.full([Nbc, 1], length_left, device = device) + torch.rand([Nbc, 1], device = device) * (length_right-length_left), torch.full([Nbc, 1], height_top, device = device)), dim=1) #top
    # 图中的边界2
    x3_boundary_one = torch.cat((torch.full([Nbc, 1], length_left, device = device) + torch.rand([Nbc, 1], device = device) * (length_right-length_left), torch.full([Nbc, 1], (thickness_mag+thickness_air), device = device)), dim=1) #middle top
    # A1与A2的交界
    x3_boundary_two = torch.cat((torch.full([Nbc, 1], length_left, device = device) + torch.rand([Nbc, 1], device = device) * (length_right-length_left), torch.full([Nbc, 1], thickness_mag, device = device)), dim=1) # middle bottom

    #边界对应的解析解
    x1_boundary_left_A1_label = exact_A1(x1_boundary_left_A1[:, 0], x1_boundary_left_A1[:, 1]).float()
    x1_boundary_left_A2_label = exact_A2(x1_boundary_left_A2[:, 0], x1_boundary_left_A2[:, 1]).float()
    x1_boundary_right_A1_label = exact_A1(x1_boundary_right_A1[:, 0], x1_boundary_right_A1[:, 1]).float()
    x1_boundary_right_A2_label = exact_A2(x1_boundary_right_A2[:, 0], x1_boundary_right_A2[:, 1]).float()
    x2_boundary_left_label = exact_A1(x2_boundary_left[:, 0], x2_boundary_left[:, 1]).float()
    x2_boundary_right_label = exact_A2(x2_boundary_right[:, 0], x2_boundary_right[:, 1]).float()
    x3_boundary_one_label = exact_A2(x3_boundary_one[:, 0], x3_boundary_one[:, 1]).float()
    x3_boundary_two_label = exact_A2(x3_boundary_two[:, 0], x3_boundary_two[:, 1]).float()

    x_bc = is_cuda(torch.cat((x1_boundary_left_A1, x1_boundary_left_A2, x1_boundary_right_A1, x1_boundary_right_A2, x2_boundary_left, x2_boundary_right, x3_boundary_one, x3_boundary_two), dim=0))
    u_bc = is_cuda(torch.cat(
        (x1_boundary_left_A1_label, x1_boundary_left_A2_label, x1_boundary_right_A1_label, x1_boundary_right_A2_label, x2_boundary_left_label, x2_boundary_right_label,x3_boundary_one_label, x3_boundary_two_label), dim=0))

    # 在求解的区域内取点,拉丁超立方在求解域内采样，N个
    x_f_N_A1 = random_fun(N, lb_A1, ub_A1)
    x_f_N_A2 = random_fun(N, lb_A2, ub_A2)
    x_f_M_A1 = random_fun(M, lb_A1, ub_A1)
    x_f_M_A2 = random_fun(M, lb_A2, ub_A2)
    x_f_A1 = torch.cat((x_f_N_A1, x_f_M_A1), dim=0)
    x_f_A2 = torch.cat((x_f_N_A2, x_f_M_A2), dim=0)
    X_train = torch.cat((x_f_A1, x_f_A2), dim=0)

    # 对坐标进行均值和std
    X_mean = torch.tensor(np.mean(np.concatenate([X_train.cpu().detach().numpy(), x1_boundary_left_A1.cpu().detach().numpy(), x1_boundary_left_A2.cpu().detach().numpy(), x1_boundary_right_A1.cpu().detach().numpy(),\
                                                 x1_boundary_right_A2.cpu().detach().numpy(), x2_boundary_left.cpu().detach().numpy(), x2_boundary_right.cpu().detach().numpy(), x3_boundary_one.cpu().detach().numpy(),\
                                                  x3_boundary_two.cpu().detach().numpy()], 0), axis=0, keepdims=True), dtype=torch.float32)

    X_std = torch.tensor(np.std(np.concatenate([X_train.cpu().detach().numpy(), x1_boundary_left_A1.cpu().detach().numpy(), x1_boundary_left_A2.cpu().detach().numpy(), x1_boundary_right_A1.cpu().detach().numpy(),\
                                                x1_boundary_right_A2.cpu().detach().numpy(), x2_boundary_left.cpu().detach().numpy(), x2_boundary_right.cpu().detach().numpy(), x3_boundary_one.cpu().detach().numpy(),\
                                                x3_boundary_two.cpu().detach().numpy()], 0), axis=0, keepdims=True), dtype=torch.float32)
    net = is_cuda(Net(layers, X_mean, X_std))

    model = Model(
        method = method,
        net=net,
        x_f_N_A1=x_f_N_A1,
        x_f_M_A1=x_f_M_A1,
        x_f_N_A2=x_f_N_A2,
        x_f_M_A2=x_f_M_A2,
        b1=x2_boundary_right,
        b2=x3_boundary_one,
        b3=x3_boundary_two,
        b4=x2_boundary_left,
        b5=x1_boundary_left_A2,
        b7=x1_boundary_right_A2,
        b6=x1_boundary_left_A1,
        b8=x1_boundary_right_A1,
        loss_b1=loss_b1,
        loss_b2=loss_b2,
        loss_b3=loss_b3,
        loss_b4=loss_b4,
        loss_b57= loss_b57,
        loss_b68=loss_b68,
        x_f_loss_fun_A1=x_f_loss_fun_A1,
        x_f_loss_fun_A2=x_f_loss_fun_A2,
        x_test=x_test,
        x_test_exact_Adx=x_test_exact_Adx,
        x_test_exact_Ady=x_test_exact_Ady,
        x_test_copper=x_test_copper_A2,
        x_test_copper_Adx=solution_copper_A2dx,
        x_test_copper_Ady=solution_copper_A2dy
    )
    model.train()   # 模型训练

    model.save(r"C:\Users\Ethan\Desktop\AMAW-PINN-master\AMAW-PINN-master\2Dpossion_equation\save")
    #print(model.x_test_estimate_collect)

