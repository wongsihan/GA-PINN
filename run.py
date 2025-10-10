#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GA-PINN 主程序入口
运行完整的训练流程，包括涡流场可视化
"""

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入并运行主程序
if __name__ == '__main__':
    from main import *
