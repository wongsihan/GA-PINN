# GA-PINN 模块化代码结构

本项目将原始的单一Python文件拆分为多个模块，以提高代码的可维护性和可读性。

## 项目结构

```
GA-PINN/
├── src/                    # 源代码目录
│   ├── utils.py           # 工具函数模块
│   ├── model.py           # 神经网络和模型类
│   ├── physics.py         # 物理方程和损失函数
│   ├── config.py          # 配置参数
│   └── main.py            # 主程序逻辑
├── results/                # 结果文件目录
│   └── *.png              # 涡流场可视化图像
├── docs/                   # 文档目录
│   ├── *.pdf              # 论文文档
│   └── tim_GAMGP-PINN-A3S150C5.py  # 原始代码
├── models/                 # 模型保存目录
├── run.py                  # 主程序入口
└── README.md              # 说明文档
```

## 模块说明

### utils.py
包含通用的工具函数：
- `random_fun()`: 拉丁超立方采样
- `is_cuda()`: GPU设备管理
- `tensor()`: 张量创建
- `loss_grad_stats()`: 梯度统计
- `loss_grad_max_mean()`: 梯度最大值和均值计算

### model.py
包含神经网络和模型类：
- `Net`: 全连接神经网络类
- `Model`: 主要的PINN模型类，包含所有训练方法
  - `run_baseline()`: 原始PINN
  - `run_AW()`: AW-PINN
  - `run_AM()`: R/GAM-PINN
  - `run_AM_AW()`: R/GAM-AW-PINN
  - `run_AM_sensors()`: 传感器方法

### physics.py
包含物理相关的函数：
- `x_f_loss_fun_A1()`: A1区域损失函数
- `x_f_loss_fun_A2()`: A2区域损失函数
- `loss_b1()` ~ `loss_b68()`: 边界损失函数
- `exact_A1()`: A1区域解析解
- `exact_A2()`: A2区域解析解

### config.py
包含所有配置参数：
- PMD物理参数
- 网络结构参数
- 训练参数
- 模型类型设置

### main.py
主程序入口，包含：
- 测试数据生成
- 边界条件设置
- 模型初始化和训练
- 结果保存

## 使用方法

### 运行完整程序
```bash
python run.py
```

### 直接运行源代码
```bash
cd src
python main.py
```

## 主要改进

1. **模块化设计**: 将单一文件拆分为功能明确的模块
2. **代码复用**: 减少重复代码，提高可维护性
3. **清晰结构**: 每个模块职责明确，便于理解和修改
4. **保持功能**: 完全保持原始代码的功能不变

## 注意事项

- 所有模块间通过import进行依赖管理
- 配置参数集中在config.py中管理
- 物理方程和损失函数独立在physics.py中
- 模型训练逻辑封装在Model类中
