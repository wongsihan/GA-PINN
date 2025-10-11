import os
import numpy as np
import torch
from torch.autograd import Variable, grad
from utils_ramaw import is_cuda, tensor, random_fun, device
from config_ramaw import *
from model_ramaw import Net, Model
from physics_ramaw import *

def main():
    print('GPU:', torch.cuda.is_available())
    print('开始训练RAMAW-PINN模型...')
    print('=' * 50)
    
    # 边界条件点生成
    # A1区域边界
    x1_boundary_left_A1 = torch.cat((torch.full([Nbc, 1], length_left, device=device), torch.full([Nbc, 1], height_bottom, device=device) + torch.rand([Nbc, 1], device=device) * thickness_mag), dim=1) #leftA1
    x1_boundary_left_A1_label = torch.cat((torch.full([Nbc, 1], length_left, device=device), torch.full([Nbc, 1], height_bottom, device=device) + torch.rand([Nbc, 1], device=device) * thickness_mag), dim=1) #leftA1
    x1_boundary_right_A1 = torch.cat((torch.full([Nbc, 1], length_right, device=device), torch.full([Nbc, 1], height_bottom, device=device) + torch.rand([Nbc, 1], device=device) * thickness_mag), dim=1) #rightA1
    x1_boundary_right_A1_label = torch.cat((torch.full([Nbc, 1], length_right, device=device), torch.full([Nbc, 1], height_bottom, device=device) + torch.rand([Nbc, 1], device=device) * thickness_mag), dim=1) #rightA1
    
    # A2区域边界
    x1_boundary_left_A2 = torch.cat((torch.full([Nbc, 1], length_left, device=device), torch.full([Nbc, 1], thickness_mag, device=device) + torch.rand([Nbc, 1], device=device) * (thickness_air + thickness_copper)), dim=1) #leftA2
    x1_boundary_left_A2_label = torch.cat((torch.full([Nbc, 1], length_left, device=device), torch.full([Nbc, 1], thickness_mag, device=device) + torch.rand([Nbc, 1], device=device) * (thickness_air + thickness_copper)), dim=1) #leftA2
    x1_boundary_right_A2 = torch.cat((torch.full([Nbc, 1], length_right, device=device), torch.full([Nbc, 1], thickness_mag, device=device) + torch.rand([Nbc, 1], device=device) * (thickness_air + thickness_copper)), dim=1) #rightA2
    x1_boundary_right_A2_label = torch.cat((torch.full([Nbc, 1], length_right, device=device), torch.full([Nbc, 1], thickness_mag, device=device) + torch.rand([Nbc, 1], device=device) * (thickness_air + thickness_copper)), dim=1) #rightA2
    
    # 上下边界
    x2_boundary_left = torch.cat((torch.full([Nbc, 1], length_left, device=device) + torch.rand([Nbc, 1], device=device) * length_right, torch.full([Nbc, 1], height_bottom, device=device)), dim=1) #bottom
    x2_boundary_left_label = torch.cat((torch.full([Nbc, 1], length_left, device=device) + torch.rand([Nbc, 1], device=device) * length_right, torch.full([Nbc, 1], height_bottom, device=device)), dim=1) #bottom
    x2_boundary_right = torch.cat((torch.full([Nbc, 1], length_left, device=device) + torch.rand([Nbc, 1], device=device) * length_right, torch.full([Nbc, 1], height_top, device=device)), dim=1) #top
    x2_boundary_right_label = torch.cat((torch.full([Nbc, 1], length_left, device=device) + torch.rand([Nbc, 1], device=device) * length_right, torch.full([Nbc, 1], height_top, device=device)), dim=1) #top
    
    # 中间边界
    x3_boundary_one = torch.cat((torch.full([Nbc, 1], length_left, device=device) + torch.rand([Nbc, 1], device=device) * length_right, torch.full([Nbc, 1], thickness_mag, device=device)), dim=1) #middle1
    x3_boundary_one_label = torch.cat((torch.full([Nbc, 1], length_left, device=device) + torch.rand([Nbc, 1], device=device) * length_right, torch.full([Nbc, 1], thickness_mag, device=device)), dim=1) #middle1
    x3_boundary_two = torch.cat((torch.full([Nbc, 1], length_left, device=device) + torch.rand([Nbc, 1], device=device) * length_right, torch.full([Nbc, 1], thickness_mag + thickness_air, device=device)), dim=1) #middle2
    x3_boundary_two_label = torch.cat((torch.full([Nbc, 1], length_left, device=device) + torch.rand([Nbc, 1], device=device) * length_right, torch.full([Nbc, 1], thickness_mag + thickness_air, device=device)), dim=1) #middle2
    
    # 边界条件标签
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
    model_path = "models/model_RAMAW-PINN.pt"
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
        x_test=None,
        x_test_exact_Adx=None,
        x_test_exact_Ady=None,
        x_test_copper=None,
        x_test_copper_Adx=None,
        x_test_copper_Ady=None
    )
    model.train()   # 模型训练

    # 保存模型
    os.makedirs("models", exist_ok=True)
    model.save("models")

if __name__ == '__main__':
    main()
