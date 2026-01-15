"""
安装CUDA版本PyTorch - 优化版本
"""
import subprocess
import sys
import os
import shutil

def check_disk_space():
    """检查可用磁盘空间"""
    drives = ['C:', 'D:', 'E:']
    for drive in drives:
        if os.path.exists(drive):
            try:
                total, used, free = shutil.disk_usage(drive)
                free_gb = free / (1024**3)
                print(f"{drive} 可用空间: {free_gb:.2f} GB")
                if free_gb > 4:  # 至少4GB空间
                    return True
            except:
                pass
    return False

def clean_pip_cache():
    """清理pip缓存释放空间"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=True)
        print("✅ pip缓存已清理")
    except:
        print("⚠️ pip缓存清理失败")

def install_cuda_pytorch():
    """安装CUDA版本的PyTorch"""
    print("🚀 安装CUDA版本PyTorch...")
    
    # 1. 清理缓存
    clean_pip_cache()
    
    # 2. 卸载CPU版本
    print("\n1. 卸载CPU版本PyTorch...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", 
            "torch", "torchvision", "torchaudio", "-y"
        ], check=True)
        print("✅ CPU版本已卸载")
    except:
        print("⚠️ 卸载失败或未安装")
    
    # 3. 安装CUDA版本
    print("\n2. 安装CUDA版本...")
    
    # 根据您的NVIDIA驱动版本560.94，支持CUDA 11.8或12.x
    cuda_commands = [
        # CUDA 11.8版本（较小，推荐）
        [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118",
            "--no-cache-dir"  # 不使用缓存减少空间需求
        ],
        # CUDA 12.1版本（备用）
        [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "--no-cache-dir"
        ],
        # 使用国内镜像（最后备用）
        [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118",
            "--trusted-host", "download.pytorch.org",
            "--no-cache-dir"
        ]
    ]
    
    for i, cmd in enumerate(cuda_commands, 1):
        try:
            print(f"\n尝试方法 {i}...")
            print("这可能需要几分钟时间，请耐心等待...")
            subprocess.run(cmd, check=True)
            print("✅ CUDA版本PyTorch安装成功!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 方法 {i} 失败: {e}")
            if i < len(cuda_commands):
                print("尝试下一种方法...")
            continue
    
    print("❌ 所有安装方法都失败了")
    return False

def verify_cuda_installation():
    """验证CUDA安装"""
    try:
        import torch
        print(f"\n🔍 验证安装结果:")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"当前GPU: {torch.cuda.get_device_name(0)}")
            
            # 测试GPU计算
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            print("✅ GPU计算测试成功!")
            return True
        else:
            print("❌ CUDA仍然不可用")
            return False
            
    except ImportError:
        print("❌ PyTorch导入失败")
        return False

def main():
    print("🔧 修复PyTorch CUDA支持...")
    
    # 检查磁盘空间
    if not check_disk_space():
        print("\n⚠️ 磁盘空间可能不足，但我们尝试继续...")
    
    # 安装CUDA版本
    if install_cuda_pytorch():
        # 验证安装
        print("\n" + "="*50)
        verify_cuda_installation()
    else:
        print("\n💡 替代方案:")
        print("1. 释放更多磁盘空间")
        print("2. 使用云端GPU服务（Google Colab）")
        print("3. 暂时使用CPU版本")

if __name__ == "__main__":
    main()