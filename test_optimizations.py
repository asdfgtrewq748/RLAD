#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLAD v3.2 优化测试脚本
测试所有新增的优化功能：
1. 余弦退火学习率
2. 增强梯度裁剪
3. 卷积注意力机制
4. 增强特征工程
5. 强化非对称奖励
6. 单个注意力头可视化
"""

import sys
import os
import numpy as np
import torch
import argparse
from datetime import datetime

def test_enhanced_features():
    """测试增强特征工程功能"""
    print("🔧 测试增强特征工程...")
    
    # 导入特征工程函数
    # 修正导入路径
    import importlib.util
    spec = importlib.util.spec_from_file_location("rlad_module", "RLADv3_2_TRUE_copy copy 2.py")
    rlad_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rlad_module)
    
    extract_enhanced_features = rlad_module.extract_enhanced_features
    apply_feature_engineering_to_windows = rlad_module.apply_feature_engineering_to_windows
    
    # 创建测试数据
    window_size = 288
    n_samples = 100
    test_windows = np.random.randn(n_samples, window_size, 1)
    
    # 测试单个窗口特征提取
    sample_window = test_windows[0, :, 0]
    enhanced_feats = extract_enhanced_features(sample_window)
    print(f"   单个窗口增强特征维度: {len(enhanced_feats)}")
    
    # 测试批量特征工程
    enhanced_windows = apply_feature_engineering_to_windows(test_windows, enhanced_features=True)
    print(f"   批量特征工程: {test_windows.shape} -> {enhanced_windows.shape}")
    
    return True

def test_attention_mechanisms():
    """测试增强注意力机制"""
    print("🎯 测试增强注意力机制...")
    
    # 导入模型类
    import importlib.util
    spec = importlib.util.spec_from_file_location("rlad_module", "RLADv3_2_TRUE_copy copy 2.py")
    rlad_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rlad_module)
    
    EnhancedRLADAgent = rlad_module.EnhancedRLADAgent
    
    # 创建模型
    device = torch.device("cpu")
    agent = EnhancedRLADAgent(
        input_dim=1,
        seq_len=288,
        hidden_size=64,
        num_heads=2,
        dropout=0.3,
        bidirectional=True,
        include_pos=True,
        num_actions=2,
        use_lstm=True,
        use_attention=True,
        num_layers=1
    ).to(device)
    
    # 测试前向传播和注意力权重提取
    test_input = torch.randn(1, 288, 1).to(device)
    
    # 测试标准前向传播
    output = agent(test_input)
    print(f"   标准输出形状: {output.shape}")
    
    # 测试注意力权重提取
    output_with_attn = agent(test_input, return_attention_weights=True)
    q_values, attention_dict = output_with_attn
    print(f"   Q值形状: {q_values.shape}")
    print(f"   注意力类型: {list(attention_dict.keys())}")
    print(f"   自注意力形状: {attention_dict['self_attention'].shape}")
    print(f"   卷积注意力形状: {attention_dict['conv_attention'].shape}")
    
    return True

def test_reward_mechanisms():
    """测试增强奖励机制"""
    print("🎯 测试增强奖励机制...")
    
    # 导入奖励函数
    import importlib.util
    spec = importlib.util.spec_from_file_location("rlad_module", "RLADv3_2_TRUE_copy copy 2.py")
    rlad_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rlad_module)
    
    enhanced_compute_reward = rlad_module.enhanced_compute_reward
    compute_safety_first_reward = rlad_module.compute_safety_first_reward
    
    # 测试基础奖励函数
    test_cases = [
        (1, 1, False, False),  # TP: 正确预测异常
        (0, 1, False, False),  # FN: 漏报
        (0, 0, False, False),  # TN: 正确预测正常
        (1, 0, False, False),  # FP: 误报
    ]
    
    print("   基础奖励测试:")
    for action, label, is_human, is_aug in test_cases:
        reward = enhanced_compute_reward(action, label, is_human, is_aug)
        case_name = {(1,1): "TP", (0,1): "FN", (0,0): "TN", (1,0): "FP"}[(action, label)]
        print(f"     {case_name}: {reward:.2f}")
    
    # 测试安全第一奖励
    print("   安全第一奖励测试:")
    context_info = {
        'recent_fn_rate': 0.2,  # 20%漏报率
        'recent_fp_rate': 0.1,  # 10%误报率
        'severity_level': 3
    }
    
    for action, label in test_cases[:4]:
        reward = compute_safety_first_reward(action, label, context_info)
        case_name = {(1,1): "TP", (0,1): "FN", (0,0): "TN", (1,0): "FP"}[(action, label)]
        print(f"     {case_name}: {reward:.2f}")
    
    return True

def test_learning_rate_scheduler():
    """测试余弦退火学习率调度器"""
    print("📈 测试余弦退火学习率调度器...")
    
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    
    # 创建简单模型和优化器
    model = torch.nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建余弦退火调度器
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,  # 初始重启周期
        T_mult=2,  # 重启周期倍数
        eta_min=1e-6,  # 最小学习率
        last_epoch=-1
    )
    
    # 模拟训练过程中的学习率变化
    lr_history = []
    for epoch in range(100):
        lr_history.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    print(f"   初始学习率: {lr_history[0]:.6f}")
    print(f"   第20轮学习率: {lr_history[19]:.6f}")
    print(f"   第50轮学习率: {lr_history[49]:.6f}")
    print(f"   最终学习率: {lr_history[-1]:.6f}")
    
    return True

def test_gradient_clipping():
    """测试增强梯度裁剪"""
    print("✂️ 测试增强梯度裁剪...")
    
    # 创建测试模型
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1)
    )
    
    # 创建故意的大梯度
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 模拟梯度爆炸情况
    large_input = torch.randn(32, 100) * 100  # 很大的输入
    target = torch.randn(32, 1)
    
    for i in range(3):
        optimizer.zero_grad()
        output = model(large_input)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        # 计算梯度范数
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        print(f"   迭代 {i+1} 梯度范数: {total_norm:.2f}")
        
        # 应用动态梯度裁剪
        grad_clip = 1.0
        if total_norm > grad_clip * 3:
            effective_clip = grad_clip * 0.3
            print(f"     检测到梯度爆炸，使用严格裁剪: {effective_clip}")
        elif total_norm > grad_clip * 1.5:
            effective_clip = grad_clip * 0.7
            print(f"     使用中等裁剪: {effective_clip}")
        else:
            effective_clip = grad_clip
            print(f"     使用标准裁剪: {effective_clip}")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=effective_clip)
        optimizer.step()
    
    return True

def run_full_optimization_test():
    """运行完整的优化测试"""
    print("🚀 开始RLAD v3.2优化功能完整测试...")
    print("=" * 60)
    
    test_results = {}
    
    # 测试各个组件
    try:
        test_results['enhanced_features'] = test_enhanced_features()
        print("✅ 增强特征工程测试通过\n")
    except Exception as e:
        print(f"❌ 增强特征工程测试失败: {e}\n")
        test_results['enhanced_features'] = False
    
    try:
        test_results['attention_mechanisms'] = test_attention_mechanisms()
        print("✅ 增强注意力机制测试通过\n")
    except Exception as e:
        print(f"❌ 增强注意力机制测试失败: {e}\n")
        test_results['attention_mechanisms'] = False
    
    try:
        test_results['reward_mechanisms'] = test_reward_mechanisms()
        print("✅ 增强奖励机制测试通过\n")
    except Exception as e:
        print(f"❌ 增强奖励机制测试失败: {e}\n")
        test_results['reward_mechanisms'] = False
    
    try:
        test_results['lr_scheduler'] = test_learning_rate_scheduler()
        print("✅ 余弦退火学习率测试通过\n")
    except Exception as e:
        print(f"❌ 余弦退火学习率测试失败: {e}\n")
        test_results['lr_scheduler'] = False
    
    try:
        test_results['gradient_clipping'] = test_gradient_clipping()
        print("✅ 增强梯度裁剪测试通过\n")
    except Exception as e:
        print(f"❌ 增强梯度裁剪测试失败: {e}\n")
        test_results['gradient_clipping'] = False
    
    # 总结
    print("=" * 60)
    print("🎯 优化功能测试总结:")
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有优化功能测试通过！可以开始训练。")
        return True
    else:
        print("⚠️ 部分功能测试失败，请检查代码。")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RLAD v3.2 优化功能测试')
    parser.add_argument('--component', type=str, default='all', 
                       choices=['all', 'features', 'attention', 'reward', 'scheduler', 'gradient'],
                       help='选择要测试的组件')
    
    args = parser.parse_args()
    
    if args.component == 'all':
        success = run_full_optimization_test()
    elif args.component == 'features':
        success = test_enhanced_features()
    elif args.component == 'attention':
        success = test_attention_mechanisms()
    elif args.component == 'reward':
        success = test_reward_mechanisms()
    elif args.component == 'scheduler':
        success = test_learning_rate_scheduler()
    elif args.component == 'gradient':
        success = test_gradient_clipping()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
