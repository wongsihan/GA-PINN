import numpy as np
import torch
from utils_ramaw import tensor

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
slip = 200
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
method = 5
N = 500     # 不移动点
M = 1000    # 移动点
Nbc = 200    #一条边界上的点
# 不同优化器迭代次数与学习率的设置
adam_iter, lbgfs_iter = 100, 10000
adam_lr, lbgfs_lr = 0.001, 0.5
# 设置模型权重调节类型
model_type = 3  # 0:baseline  1:AW  2:AM  3:AM_AW  4:AM_sensors
# 采样方式设置
AM_type = 0  # 0:RAM  1:WAM  #2:None
AM_K = 1
AM_count = 20
AW_lr = 0.001
pic_name = "RAMAW-PINN-A3S200C5"
model_name = "model_RAMAW-A3S200C5-PINN.pt"

