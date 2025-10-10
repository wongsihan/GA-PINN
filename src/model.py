import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import trange
from torch.autograd import Variable, grad
from scipy import integrate
from utils import is_cuda, tensor, loss_grad_stats, loss_grad_max_mean

# 全连接神经网络
class Net(nn.Module):
    def __init__(self, layers, mean=0, std=1):
        super(Net, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.mean = mean
        self.std = std
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)
    
    # 前向传播
    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = (x - self.mean.to(device)) / self.std.to(device)
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)     # (1500,2)
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
        a = self.linear[-1](a)
        return a

class Model:
    def __init__(self, net, b1, b2, b3, b4, b5, b7, b6, b8, loss_b1, loss_b2, loss_b3, loss_b4, loss_b57, loss_b68, 
                 x_f_loss_fun_A1, x_f_loss_fun_A2, x_test, x_test_exact_Adx, x_test_exact_Ady, 
                 x_f_N_A1, x_f_M_A1, x_f_N_A2, x_f_M_A2, method, x_test_copper, x_test_copper_Adx, x_test_copper_Ady):
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
    
    def cal_exact_torque(self, x1, x2):
        from config import length_left, length_right, field_res, thickness_mag, thickness_air, thickness_copper, sigma, omega, Rm, L
        from physics import exact_A2
        
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
        from config import length_left, length_right, field_res, thickness_mag, thickness_air, thickness_copper, sigma, omega, Rm, L
        from physics import exact_A2
        
        # draw field
        x_draw, y_draw = np.mgrid[length_left:length_right:field_res * 1j,
                         thickness_mag + thickness_air:thickness_mag + thickness_air + thickness_copper:field_res * 1j]
        
        xy_draw = np.c_[x_draw.ravel(), y_draw.ravel()]
        xy_draw = tensor(xy_draw, requires_grad=True)
        # 磁矢势A
        xy_draw = is_cuda(xy_draw)
        u = self.train_U(xy_draw)

        x1 = xy_draw[:, [0]]
        x2 = xy_draw[:, [1]]
        
        new_u = u.reshape(x_draw.shape + (2,))
        Adx = new_u[..., 0]
        Ady = new_u[..., 1]
        eddy_current = -sigma * omega * Rm * Adx
        
        J2 = eddy_current * eddy_current
        torque = integrate.simps(J2.T.cpu().detach(), y_draw.T, axis=0)
        torque = integrate.simps(torque, x_draw[:, 0])
        torque = (torque * 10 * L) / (sigma * omega)
        print("Torque is:", torque)
        exact_torque = self.cal_exact_torque(x1, x2)
        torque_residual = abs(exact_torque - torque)
        print("torque_residual is:", torque_residual)
        
        plt.figure(figsize=(20, 4))
        plt.subplot(1, 2, 1)
        plt.contour(x_draw, y_draw,
                    eddy_current[:, :].cpu().detach().numpy(), 10000, cmap='jet', zorder=1)
        plt.colorbar()
        plt.xlabel('x', fontsize=14, fontname='Times New Roman')
        plt.ylabel('y', fontsize=14, fontname='Times New Roman')

        Bz = -Adx.cpu().detach().numpy()
        Bx = -Ady.cpu().detach().numpy()
        plt.subplot(1, 2, 2)
        M2 = np.hypot(Bx, Bz)
        plt.quiver(-x_draw, y_draw, Bx, Bz, M2, width=0.005,
                   scale=60, cmap='jet')
        plt.colorbar()
        plt.xlabel('x', fontsize=14, fontname='Times New Roman')
        plt.ylabel('y', fontsize=14, fontname='Times New Roman')
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

    def render_residual(self, move_count, pic_name='residual'):
        from config import length_left, length_right, field_res, thickness_mag, thickness_air, thickness_copper
        from physics import exact_A2
        
        x_draw, y_draw = np.mgrid[length_left:length_right:field_res * 1j,
                         thickness_mag + thickness_air:thickness_mag + thickness_air + thickness_copper:field_res * 1j]

        xy_draw = np.c_[x_draw.ravel(), y_draw.ravel()]
        xy_draw = tensor(xy_draw, requires_grad=True)
        # 磁矢势A
        xy_draw = is_cuda(xy_draw)
        u = self.train_U(xy_draw)  # 400，2

        x1 = xy_draw[:, [0]]  # 400，1
        x2 = xy_draw[:, [1]]

        thisA2 = exact_A2(x1, x2)  # 400，1
        thisA2dx = grad(thisA2, x1, grad_outputs=torch.ones_like(thisA2), create_graph=True)
        thisA2dx = thisA2dx[0]  # 400，1
        thisA2dy = grad(thisA2, x2, grad_outputs=torch.ones_like(thisA2), create_graph=True)
        thisA2dy = thisA2dy[0]
        Adx_exact = thisA2dx.reshape(field_res, field_res)  # 20,20
        Ady_exact = thisA2dy.reshape(field_res, field_res)

        new_u = u.reshape(x_draw.shape + (2,))
        Adx_pred = new_u[..., 0]
        Ady_pred = new_u[..., 1]

        Adx = abs(Adx_exact - Adx_pred)
        Ady = abs(Ady_exact - Ady_pred)

        plt.figure(figsize=(20, 4))
        plt.subplot(1, 2, 1)
        plt.contourf(x_draw, y_draw,
                    Adx[:, :].cpu().detach().numpy(), 10000, cmap='jet', zorder=1)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.contourf(x_draw, y_draw,
                    Ady[:, :].cpu().detach().numpy(), 10000, cmap='jet', zorder=1)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
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
        
        # 区域内物理值预测，数量级是一样的吗
        loss_equation_A1 = (self.x_f_loss_fun_A1(x_f_A1, self.train_U).sum())
        loss_equation_A2 = (self.x_f_loss_fun_A2(x_f_A2, self.train_U).sum())
        # 区域内的损失函数
        loss_equation = loss_equation_A1 + loss_equation_A2
        # 边界物理值预测与标签的均方差，需要修改,考虑到x_f_loss_fun_Ax中具有导数项，故将边界的损失函数写入到区域内计算
        loss_val_b1 = (self.loss_b1(self.b1, self.train_U))  # dAb1/dy
        loss_val_b2 = (self.loss_b2(self.b2, self.train_U))
        loss_val_b3 = (self.loss_b3(self.b3, self.train_U))
        loss_val_b4 = (self.loss_b4(self.b4, self.train_U))
        loss_val_b57 = (self.loss_b57(self.b5, self.b7, self.train_U))
        loss_val_b68 = (self.loss_b68(self.b6, self.b8, self.train_U))
        # 边界上的损失函数
        loss_label = (loss_val_b1 + loss_val_b2 + loss_val_b3 + loss_val_b4 + loss_val_b57 + loss_val_b68)

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

        # 这里的残差采用了平方后取均值，原始的实验表明可能平方后求和的loss更有效
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
        l_bc = (loss_val_b1 + loss_val_b2 + loss_val_b3 + loss_val_b4 + loss_val_b57 + loss_val_b68)

        with torch.no_grad():
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
            return l_reg, lambd * l_bc, lambd, l_bc

    # computer backward loss
    def LBGFS_epoch_loss(self):
        self.optimizer_LBGFS.zero_grad()
        x_f_A1 = torch.cat((self.x_f_N_A1, self.x_f_M_A1), dim=0)
        x_f_A2 = torch.cat((self.x_f_N_A2, self.x_f_M_A2), dim=0)

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
        loss_label = (loss_val_b1 + loss_val_b2 + loss_val_b3 + loss_val_b4 + loss_val_b57 + loss_val_b68)

        if self.start_loss_collect:
            self.x_f_loss_collect.append([self.net.iter, loss_equation.item()])
            self.x_label_loss_collect.append([self.net.iter, loss_label.item()])

        loss = self.true_loss(loss_equation, loss_label)
        loss.backward(retain_graph=True)
        self.net.iter += 1
        return loss

    def evaluate(self):
        self.x_cooper_test = is_cuda(self.x_test_copper)
        pred_cop = self.train_U(self.x_cooper_test).cpu().detach().numpy()
        exact_cop = torch.cat((self.x_test_copper_Adx, self.x_test_copper_Ady), axis=1).cpu().detach().numpy()
        error = np.linalg.norm(pred_cop - exact_cop, 2) / np.linalg.norm(exact_cop, 2)
        return error

    # 原始PINN，无移动点、无自适应loss权重
    def run_baseline(self):
        from config import AM_count, lbgfs_lr, lbgfs_iter, adam_lr, adam_iter, pic_name
        from utils import random_fun
        
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
                                  'Loss': '{0:.2e}'.format(loss.item())
                                  })

            print('Adam done!')
            error = self.evaluate()
            print('change_counts', move_count, 'Test_L2error:', '{0:.2e}'.format(error))
            self.x_test_estimate_collect.append([move_count, '{0:.2e}'.format(error)])
            self.render_res(move_count, pic_name)

    # AW-PINN，无移动点、有自适应loss权重
    def run_AW(self):
        from config import AM_count, lbgfs_lr, lbgfs_iter, adam_lr, adam_iter, AW_lr, pic_name
        
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

    # R/GAM-PINN，有残差or梯度移动点、无自适应loss权重
    def run_AM(self):
        from config import AM_count, lbgfs_lr, lbgfs_iter, adam_lr, adam_iter, AW_lr, pic_name, AM_type, AM_K, M, lb_A1, ub_A1, lb_A2, ub_A2
        from utils import random_fun
        
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
                print('One loop done!')

            self.render_res(move_count, pic_name)

    # R/GAM-AW-PINN，有残差or梯度移动点、有自适应loss权重
    def run_AM_AW(self):
        from config import AM_count, lbgfs_lr, lbgfs_iter, adam_lr, adam_iter, AW_lr, pic_name, AM_type, AM_K, M, lb_A1, ub_A1, lb_A2, ub_A2
        from utils import random_fun
        
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
                print('One loop done!')

            self.render_res(move_count, pic_name)

    def run_AM_sensors(self):
        from config import AM_count, lbgfs_lr, lbgfs_iter, adam_lr, adam_iter, AW_lr, pic_name, AM_type, AM_K, M, lb_A1, ub_A1, lb_A2, ub_A2
        from utils import random_fun
        
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
                loss_e, loss_label, lambd, loss_boundary = self.custom_train(self.method)
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
            print('change_counts', move_count, 'Test_L2error:', '{0:.2e}'.format(error), 'Loss-pde',
                  '{0:.2e}'.format(loss_e.item()),
                  'Loss-label', '{0:.2e}'.format(loss_label.item()),
                  'Loss-boundary', '{0:.2e}'.format(loss_boundary.item()),
                  'lambd', '{0:.2e}'.format(lambd.item()))
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
            self.render_residual(move_count, pic_name)

    def train(self):
        from config import model_type
        
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
        import time
        import os
        from config import model_name
        
        timestr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        modelfname = os.path.join(dirname, model_name + timestr)
        torch.save(self.net, modelfname)
