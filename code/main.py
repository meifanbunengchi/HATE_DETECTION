import torch


def check_device():
    # 检查CUDA（NVIDIA GPU）是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 获取GPU设备名称
        gpu_name = torch.cuda.get_device_name(0)
        print(f"当前使用GPU: {gpu_name}")
        print(f"设备索引: {torch.cuda.current_device()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    # 检查MPS（Apple Silicon GPU）是否可用
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("当前使用Apple Silicon GPU (MPS)")
    # 否则使用CPU
    else:
        device = torch.device("cpu")
        print("当前使用CPU")

    return device


# 执行设备检查
current_device = check_device()
print(f"PyTorch最终使用的设备: {current_device}")