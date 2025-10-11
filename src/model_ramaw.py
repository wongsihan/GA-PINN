import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable, grad
from tqdm import tqdm, trange
from scipy import integrate
from utils_ramaw import is_cuda, tensor, device
from config_ramaw import length_left, length_right, field_res, thickness_mag, thickness_air, thickness_copper, sigma, omega, Rm, L, model_type, AM_count, lbgfs_lr, adam_lr, adam_iter, lbgfs_iter, AW_lr, AM_type, L
from physics_ramaw import exact_A2, draw_Adx_exact_points, draw_Ady_exact_points

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
            b = grad(loss, m.bias, retain_graph=True)[0]
            grad_ = torch.cat((grad_, w.view(-1), b))
    # collect gradient statistics
    diffs = grad_ - torch.mean(grad_)
    std = torch.std(grad_)
    zscores = diffs / std
    # skews = torch.mean(torch.pow(zscores, 3.0))
    kurt = torch.mean(torch.pow(zscores, 4.0))
    return std, kurt

def loss_grad_max_mean(loss, net, lambg=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    # collect gradients
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        if (m == 0):
            w = torch.abs(lambg * grad(loss, m.weight, retain_graph=True)[0])
            b = torch.abs(lambg * grad(loss, m.bias, retain_graph=True)[0])
            grad_ = torch.cat((w.view(-1), b))
        else:
            w = torch.abs(lambg * grad(loss, m.weight, retain_graph=True)[0])
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
        self.x_label_s = torch.tensor(1.0, requires_grad=True, device=device)
        self.x_f_s = torch.tensor(1.0, requires_grad=True, device=device)
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
        thisA2 = exact_A2(xy_draw[:, [0]], xy_draw[:, [1]])
        thisA2dx = grad(thisA2, xy_draw, grad_outputs=torch.ones_like(thisA2), create_graph=True)
        thisA2dx = thisA2dx[0][:, 0]
        thisres = thisA2dx.reshape(field_res, field_res)
        eddy_current = -sigma * omega * Rm * thisres
        J2 = eddy_current * eddy_current
        torque = integrate.simps(J2.T.cpu().detach().numpy(), y_draw.T, axis=0)
        torque = integrate.simps(torque, x_draw[:, 0])
        torque = (torque * 10 * L) / (sigma * omega)
        return torque

    def render_res(self, move_count, pic_name):
        from config_ramaw import height_bottom, height_top
        x_draw = torch.linspace(length_left, length_right, field_res).view(-1, 1).repeat(1, field_res).view(-1, 1)
        y_draw = torch.linspace(height_bottom, height_top, field_res).view(1, -1).repeat(field_res, 1).view(-1, 1)
        x_draw = is_cuda(x_draw)
        y_draw = is_cuda(y_draw)
        xy_draw = torch.cat((x_draw, y_draw), dim=1)
        xy_draw = is_cuda(xy_draw)
        u = self.train_U(xy_draw)
        
        x1 = xy_draw[:, [0]]
        x2 = xy_draw[:, [1]]
        
        new_u = u.reshape(field_res, field_res, 2)
        Adx = new_u[..., 0]
        Ady = new_u[..., 1]
        eddy_current = -sigma * omega * Rm * Adx
        
        # 准备绘图数据
        x_draw_2d = x_draw.reshape(field_res, field_res).cpu().detach().numpy()
        y_draw_2d = y_draw.reshape(field_res, field_res).cpu().detach().numpy()
        
        J2 = eddy_current * eddy_current
        torque = integrate.simps(J2.cpu().detach().numpy(), y_draw_2d, axis=0)
        torque = integrate.simps(torque, x_draw_2d[:, 0])
        torque = (torque * 10 * L) / (sigma * omega)
        print("Torque is:", torque)
        exact_torque = self.cal_exact_torque(x1, x2)
        torque_residual = abs(exact_torque - torque)
        print("torque_residual is:", torque_residual)
        
        import os
        os.makedirs("results/ramaw", exist_ok=True)
        filename = f"results/ramaw/{pic_name}1-{move_count}.png"
        filename2 = f"results/ramaw/{pic_name}2-{move_count}.png"

        plt.figure(figsize=(20, 4))
        plt.subplot(1, 2, 1)
        plt.contour(x_draw_2d, y_draw_2d,
                    Adx.cpu().detach().numpy(), 10000, cmap='jet', zorder=1)
        plt.colorbar()
        plt.xlabel('x', fontsize=14, fontname='Times New Roman')
        plt.ylabel('y', fontsize=14, fontname='Times New Roman')

        Bz = -Adx.cpu().detach().numpy()
        Bx = -Ady.cpu().detach().numpy()
        plt.subplot(1, 2, 2)
        M2 = np.hypot(Bx, Bz)
        plt.contour(x_draw_2d, y_draw_2d,
                    Ady.cpu().detach().numpy(), 10000, cmap='jet', zorder=1)
        plt.colorbar()
        plt.xlabel('x', fontsize=14, fontname='Times New Roman')
        plt.ylabel('y', fontsize=14, fontname='Times New Roman')
        
        plt.title(filename)
        plt.savefig(filename)
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.plot(x1[:, [0]].cpu().detach().numpy(), x2[:, [0]].cpu().detach().numpy(),
                 u[:, 0].cpu().detach().numpy().reshape(-1, 1))

        plt.title(filename2)
        plt.savefig(filename2)
        plt.close()

    def render_residual(self, move_count, pic_name):
        from config_ramaw import height_bottom, height_top
        x_draw = torch.linspace(length_left, length_right, field_res).view(-1, 1).repeat(1, field_res).view(-1, 1)
        y_draw = torch.linspace(height_bottom, height_top, field_res).view(1, -1).repeat(field_res, 1).view(-1, 1)
        x_draw = is_cuda(x_draw)
        y_draw = is_cuda(y_draw)
        xy_draw = torch.cat((x_draw, y_draw), dim=1)
        xy_draw = is_cuda(xy_draw)
        u = self.train_U(xy_draw)
        
        x1 = xy_draw[:, [0]]
        x2 = xy_draw[:, [1]]
        
        new_u = u.reshape(field_res, field_res, 2)
        Adx = new_u[..., 0]
        Ady = new_u[..., 1]
        
        # 准备绘图数据
        x_draw_2d = x_draw.reshape(field_res, field_res).cpu().detach().numpy()
        y_draw_2d = y_draw.reshape(field_res, field_res).cpu().detach().numpy()
        
        import os
        os.makedirs("results/ramaw", exist_ok=True)
        filename = f"results/ramaw/{pic_name}1-{move_count}-residual.png"
        filename2 = f"results/ramaw/{pic_name}2-{move_count}-residual.png"

        plt.title(filename)
        plt.savefig(filename)
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.plot(x1[:, [0]].cpu().detach().numpy(), x2[:, [0]].cpu().detach().numpy(),
                 u[:, 0].cpu().detach().numpy().reshape(-1, 1))

        plt.title(filename2)
        plt.savefig(filename2)
        plt.close()

    def train(self):
        start_time = time.time()
        print('开始训练RAMAW-PINN模型...')
        print('=' * 50)
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

    def run_baseline(self):
        from config_ramaw import adam_iter, adam_lr
        self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                 max_iter=lbgfs_iter)
        optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)

        pbar = trange(adam_iter, ncols=100)
        for i in pbar:
            self.s_collect.append([self.net.iter, self.x_f_s.item(), self.x_label_s.item()])
            optimizer_adam.zero_grad()
            loss = self.loss_function()
            loss.backward(retain_graph=True)
            optimizer_adam.step()
            if i % 100 == 0:
                print(f'Iter {i}, Loss: {loss.item():.2e}')

    def run_AW(self):
        from config_ramaw import adam_iter, adam_lr, AW_lr
        self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                 max_iter=lbgfs_iter)

        optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
        optimizer_adam_weight = torch.optim.Adam([self.x_f_s] + [self.x_label_s],
                                                 lr=AW_lr)

        pbar = trange(adam_iter, ncols=100)
        for i in pbar:
            self.s_collect.append([self.net.iter, self.x_f_s.item(), self.x_label_s.item()])
            optimizer_adam.zero_grad()
            optimizer_adam_weight.zero_grad()
            loss = self.loss_function()
            loss.backward(retain_graph=True)
            optimizer_adam.step()
            optimizer_adam_weight.step()
            if i % 100 == 0:
                print(f'Iter {i}, Loss: {loss.item():.2e}')

    def run_AM(self):
        from config_ramaw import adam_iter, adam_lr, AM_count, lb_A1, ub_A1, lb_A2, ub_A2
        from utils_ramaw import random_fun
        self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                 max_iter=lbgfs_iter)

        optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
        optimizer_adam_weight = torch.optim.Adam([self.x_f_s] + [self.x_label_s],
                                                 lr=AW_lr)

        pbar = trange(adam_iter, ncols=100)
        for i in pbar:
            self.s_collect.append([self.net.iter, self.x_f_s.item(), self.x_label_s.item()])
            optimizer_adam.zero_grad()
            optimizer_adam_weight.zero_grad()
            loss = self.loss_function()
            loss.backward(retain_graph=True)
            optimizer_adam.step()
            optimizer_adam_weight.step()
            if i % 100 == 0:
                print(f'Iter {i}, Loss: {loss.item():.2e}')
        
        self.render_res(0, 'RAMAW-PINN')
        self.render_residual(0, 'RAMAW-PINN')

    def run_AM_AW(self):
        from config_ramaw import adam_iter, adam_lr, AM_count, lb_A1, ub_A1, lb_A2, ub_A2
        from utils_ramaw import random_fun
        self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                 max_iter=lbgfs_iter)

        optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
        optimizer_adam_weight = torch.optim.Adam([self.x_f_s] + [self.x_label_s],
                                                 lr=AW_lr)

        pbar = trange(adam_iter, ncols=100)
        for i in pbar:
            self.s_collect.append([self.net.iter, self.x_f_s.item(), self.x_label_s.item()])
            optimizer_adam.zero_grad()
            optimizer_adam_weight.zero_grad()
            loss = self.loss_function()
            loss.backward(retain_graph=True)
            optimizer_adam.step()
            optimizer_adam_weight.step()
            if i % 100 == 0:
                print(f'Iter {i}, Loss: {loss.item():.2e}')
        
        self.render_res(0, 'RAMAW-PINN')
        self.render_residual(0, 'RAMAW-PINN')

    def run_AM_sensors(self):
        from config_ramaw import adam_iter, adam_lr, AM_count
        self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                 max_iter=lbgfs_iter)

        optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
        optimizer_adam_weight = torch.optim.Adam([self.x_f_s] + [self.x_label_s],
                                                 lr=AW_lr)

        pbar = trange(adam_iter, ncols=100)
        for i in pbar:
            self.s_collect.append([self.net.iter, self.x_f_s.item(), self.x_label_s.item()])
            optimizer_adam.zero_grad()
            optimizer_adam_weight.zero_grad()
            loss = self.loss_function()
            loss.backward(retain_graph=True)
            optimizer_adam.step()
            optimizer_adam_weight.step()
            if i % 100 == 0:
                print(f'Iter {i}, Loss: {loss.item():.2e}')
        
        self.render_res(0, 'RAMAW-PINN')
        self.render_residual(0, 'RAMAW-PINN')

    def loss_function(self):
        # 域内损失
        l_reg_A1 = torch.mean(self.x_f_loss_fun_A1(self.x_f_N_A1, self.net))
        l_reg_A2 = torch.mean(self.x_f_loss_fun_A2(self.x_f_N_A2, self.net))
        
        # 边界损失
        l_b1 = self.loss_b1(self.b1, self.net)
        l_b2 = self.loss_b2(self.b2, self.net)
        l_b3 = self.loss_b3(self.b3, self.net)
        l_b4 = self.loss_b4(self.b4, self.net)
        l_b57 = self.loss_b57(self.b5, self.b7, self.net)
        l_b68 = self.loss_b68(self.b6, self.b8, self.net)
        
        # 总损失
        loss = l_reg_A1 + l_reg_A2 + l_b1 + l_b2 + l_b3 + l_b4 + l_b57 + l_b68
        return loss

    def save(self, dirname):
        timestr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        modelfname = os.path.join(dirname, "model_" + timestr)
        torch.save(self.net, modelfname)
