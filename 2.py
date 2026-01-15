"""
安装支持GPU的PyTorch
"""
import subprocess
import sys

def install_pytorch_cuda():
    print("🚀 安装支持CUDA的PyTorch...")
    
    # 先卸载现有的PyTorch
    print("1. 卸载现有PyTorch...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])
    except:
        print("   没有发现现有PyTorch安装")
    
    # 安装CUDA版本的PyTorch
    print("2. 安装CUDA版本的PyTorch...")
    cuda_command = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ]
    
    try:
        subprocess.check_call(cuda_command)
        print("✅ PyTorch CUDA版本安装成功")
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        print("尝试使用国内镜像源...")
        
        # 备用方案：使用清华镜像
        backup_command = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio",
            "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
        ]
        
        try:
            subprocess.check_call(backup_command)
            print("✅ 使用镜像源安装成功")
        except:
            print("❌ 所有安装方式都失败了")

if __name__ == "__main__":
    install_pytorch_cuda()