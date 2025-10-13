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
    # 导入主程序模块
    from main import main
    
    # 执行主程序
    print("开始运行GA-PINN训练程序...")
    print("=" * 50)
    main()

