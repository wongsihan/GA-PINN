import os
import torch
import numpy as np
from torch.autograd import Variable, grad
from utils_lb import is_cuda, tensor, random_fun, device
from model_lb import Net, Model
from physics_lb import x_f_loss_fun_A1, x_f_loss_fun_A2, loss_b1, loss_b2, loss_b3, loss_b4, loss_b57, loss_b68, exact_A1, exact_A2
from config_lb import length_left, length_right, field_res, thickness_mag, thickness_air, thickness_copper, sigma, omega, Rm, u0, height_bottom, height_top, Nbc, N, M, lb_A1, ub_A1, lb_A2, ub_A2, layers, method

def main():
    """主程序函数"""
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
    x1_boundary_left_A1 = torch.cat((torch.full([Nbc, 1], length_left, device=device), torch.full([Nbc, 1], height_bottom, device=device) + torch.rand([Nbc, 1], device=device) * thickness_mag), dim=1) #leftA1
    # A2左侧
    x1_boundary_left_A2 = torch.cat((torch.full([Nbc, 1], length_left, device=device), torch.full([Nbc, 1], thickness_mag, device=device) + torch.rand([Nbc, 1], device=device) * (thickness_air+thickness_copper)), dim=1) #leftA2
    # A1右侧
    x1_boundary_right_A1 = torch.cat((torch.full([Nbc, 1], length_right, device=device), torch.full([Nbc, 1], height_bottom, device=device) + torch.rand([Nbc, 1], device=device) * thickness_mag), dim=1) #rightA1
    # A2右侧
    x1_boundary_right_A2 = torch.cat((torch.full([Nbc, 1], length_right, device=device), torch.full([Nbc, 1], thickness_mag, device=device) + torch.rand([Nbc, 1], device=device) * (thickness_air+thickness_copper)), dim=1) #rightA2
    # A1下侧
    x2_boundary_left = torch.cat((torch.full([Nbc, 1], length_left, device=device) + torch.rand([Nbc, 1], device=device) * (length_right-length_left), torch.full([Nbc, 1], height_bottom, device=device)), dim=1) #bottom
    # A2上侧
    x2_boundary_right = torch.cat((torch.full([Nbc, 1], length_left, device=device) + torch.rand([Nbc, 1], device=device) * (length_right-length_left), torch.full([Nbc, 1], height_top, device=device)), dim=1) #top
    # 图中的边界2
    x3_boundary_one = torch.cat((torch.full([Nbc, 1], length_left, device=device) + torch.rand([Nbc, 1], device=device) * (length_right-length_left), torch.full([Nbc, 1], (thickness_mag+thickness_air), device=device)), dim=1) #middle top
    # A1与A2的交界
    x3_boundary_two = torch.cat((torch.full([Nbc, 1], length_left, device=device) + torch.rand([Nbc, 1], device=device) * (length_right-length_left), torch.full([Nbc, 1], thickness_mag, device=device)), dim=1) # middle bottom

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

    # 检查是否有预训练模型
    model_path = "models/model_LB-PINN.pt"
    if os.path.exists(model_path):
        try:
            net = torch.load(model_path, map_location=device)
            print("load previous model success")
        except:
            print("failed to load previous model, using new model")
    else:
        print("no previous model found, using new model")

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

    model.save(r"D:\文稿\发表论文\2024MSSP\实验数据\20241230\save")

if __name__ == '__main__':
    main()

