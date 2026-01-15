"""
模型稳定性训练监控器
用于监控训练过程中的损失、梯度和性能指标
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import warnings

class TrainingStabilityMonitor:
    def __init__(self, window_size=50, loss_threshold=0.5, gradient_threshold=1.0):
        """
        初始化训练稳定性监控器
        
        Args:
            window_size: 滑动窗口大小
            loss_threshold: 损失阈值
            gradient_threshold: 梯度阈值
        """
        self.window_size = window_size
        self.loss_threshold = loss_threshold
        self.gradient_threshold = gradient_threshold
        
        # 监控指标
        self.losses = deque(maxlen=window_size)
        self.gradients = deque(maxlen=window_size)
        self.performance_metrics = deque(maxlen=window_size)
        
        # 稳定性指标
        self.loss_variance = 0.0
        self.gradient_variance = 0.0
        self.performance_variance = 0.0
        
        # 警告计数
        self.warning_count = 0
        self.instability_episodes = []
        
    def update(self, loss, gradient_norm, performance_metric, episode):
        """更新监控指标"""
        self.losses.append(float(loss))
        self.gradients.append(float(gradient_norm))
        self.performance_metrics.append(float(performance_metric))
        
        # 计算方差
        if len(self.losses) >= 10:
            self.loss_variance = np.var(list(self.losses))
            self.gradient_variance = np.var(list(self.gradients))
            self.performance_variance = np.var(list(self.performance_metrics))
            
            # 检查稳定性
            self._check_stability(episode)
    
    def _check_stability(self, episode):
        """检查训练稳定性"""
        recent_losses = list(self.losses)[-10:]
        recent_gradients = list(self.gradients)[-10:]
        recent_performance = list(self.performance_metrics)[-10:]
        
        # 检查损失稳定性
        if np.mean(recent_losses) > self.loss_threshold:
            self.warning_count += 1
            print(f"⚠️ Episode {episode}: 损失过高 (平均: {np.mean(recent_losses):.4f})")
            
        # 检查梯度稳定性
        if np.mean(recent_gradients) > self.gradient_threshold:
            self.warning_count += 1
            print(f"⚠️ Episode {episode}: 梯度过大 (平均: {np.mean(recent_gradients):.4f})")
            
        # 检查性能波动
        if len(recent_performance) >= 5 and np.std(recent_performance) > 0.1:
            self.warning_count += 1
            print(f"⚠️ Episode {episode}: 性能不稳定 (标准差: {np.std(recent_performance):.4f})")
            self.instability_episodes.append(episode)
    
    def get_stability_report(self):
        """获取稳定性报告"""
        if len(self.losses) < 10:
            return "训练数据不足，无法生成稳定性报告"
        
        report = f"""
📊 训练稳定性报告:
{'='*50}
损失统计:
  - 平均损失: {np.mean(self.losses):.4f}
  - 损失方差: {self.loss_variance:.6f}
  - 最近10次平均: {np.mean(list(self.losses)[-10:]):.4f}

梯度统计:
  - 平均梯度范数: {np.mean(self.gradients):.4f}
  - 梯度方差: {self.gradient_variance:.6f}
  - 最近10次平均: {np.mean(list(self.gradients)[-10:]):.4f}

性能统计:
  - 平均性能: {np.mean(self.performance_metrics):.4f}
  - 性能方差: {self.performance_variance:.6f}
  - 最近10次平均: {np.mean(list(self.performance_metrics)[-10:]):.4f}

稳定性评估:
  - 警告次数: {self.warning_count}
  - 不稳定episode: {self.instability_episodes}
  - 总体稳定性: {'良好' if self.warning_count < 5 else '需要关注' if self.warning_count < 15 else '不稳定'}
"""
        
        return report
    
    def suggest_optimizations(self):
        """建议优化措施"""
        suggestions = []
        
        if np.mean(self.losses) > 0.5:
            suggestions.append("建议降低学习率或增加正则化")
        
        if self.loss_variance > 0.1:
            suggestions.append("建议增大批次大小或使用更温和的学习率调度")
            
        if np.mean(self.gradients) > 1.0:
            suggestions.append("建议降低梯度裁剪阈值")
            
        if self.performance_variance > 0.02:
            suggestions.append("建议增加模型容量或改善数据质量")
            
        if len(self.instability_episodes) > len(self.losses) * 0.2:
            suggestions.append("建议重新设计网络架构或使用更稳定的损失函数")
        
        return suggestions if suggestions else ["当前训练稳定性良好"]
    
    def plot_training_curves(self, save_path=None):
        """绘制训练曲线"""
        if len(self.losses) < 5:
            print("数据不足，无法绘制曲线")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0,0].plot(self.losses, 'b-', alpha=0.7)
        axes[0,0].axhline(y=self.loss_threshold, color='r', linestyle='--', 
                         label=f'阈值: {self.loss_threshold}')
        axes[0,0].set_title('训练损失')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # 梯度范数曲线
        axes[0,1].plot(self.gradients, 'g-', alpha=0.7)
        axes[0,1].axhline(y=self.gradient_threshold, color='r', linestyle='--',
                         label=f'阈值: {self.gradient_threshold}')
        axes[0,1].set_title('梯度范数')
        axes[0,1].set_ylabel('Gradient Norm')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # 性能指标曲线
        axes[1,0].plot(self.performance_metrics, 'm-', alpha=0.7)
        axes[1,0].set_title('性能指标')
        axes[1,0].set_ylabel('Performance')
        axes[1,0].grid(True)
        
        # 稳定性统计
        axes[1,1].bar(['损失方差', '梯度方差', '性能方差'], 
                     [self.loss_variance, self.gradient_variance, self.performance_variance])
        axes[1,1].set_title('稳定性统计')
        axes[1,1].set_ylabel('方差')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()

# 使用示例
def integrate_stability_monitor():
    """
    在主训练循环中集成稳定性监控器的示例
    """
    monitor = TrainingStabilityMonitor(
        window_size=50,
        loss_threshold=0.3,  # 降低损失阈值
        gradient_threshold=0.5  # 降低梯度阈值
    )
    
    return monitor
