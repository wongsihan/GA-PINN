#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一运行脚本 - 运行所有重构后的PINN程序
"""

import sys
import os
import subprocess

def run_program(program_name, script_name):
    """运行指定的程序"""
    print(f"\n{'='*60}")
    print(f"开始运行 {program_name}")
    print(f"{'='*60}")
    
    try:
        # 运行程序
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"[OK] {program_name} 运行成功")
            print("输出:")
            print(result.stdout)
        else:
            print(f"[ERROR] {program_name} 运行失败")
            print("错误信息:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {program_name} 运行超时")
    except Exception as e:
        print(f"[ERROR] {program_name} 运行出错: {e}")

def main():
    """主函数"""
    print("开始运行所有重构后的PINN程序")
    print("=" * 60)
    
    # 程序列表
    programs = [
        ("GA-PINN", "run_ga.py"),
        ("IDW-PINN", "run_idw.py"), 
        ("LB-PINN", "run_lb.py"),
        ("PMC-PINN", "run_pmc.py"),
        ("RAMAW-PINN", "run_ramaw.py")
    ]
    
    # 检查文件是否存在
    missing_files = []
    for name, script in programs:
        if not os.path.exists(script):
            missing_files.append((name, script))
    
    if missing_files:
        print("[ERROR] 以下文件不存在:")
        for name, script in missing_files:
            print(f"   - {script} ({name})")
        print("\n请确保所有重构文件都已创建完成。")
        return
    
    # 运行所有程序
    for name, script in programs:
        run_program(name, script)
        print(f"\n{name} 运行完成\n")
    
    print("[SUCCESS] 所有程序运行完成！")
    print("结果图片保存在 results/ 目录中")

if __name__ == '__main__':
    main()


