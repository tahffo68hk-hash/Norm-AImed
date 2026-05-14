import torch
import subprocess
import os

def run_diag():
    print('--- 1. 物理层 (nvidia-smi) ---')
    try:
        print(subprocess.check_output(['nvidia-smi']).decode())
    except:
        print('nvidia-smi failed')

    print('--- 2. 系统层 (nvcc) ---')
    try:
        print(subprocess.check_output(['nvcc', '--version']).decode())
    except:
        # Try finding in PATH if command fails
        print('nvcc not found or failed')
        path = os.environ.get('PATH', '')
        if 'CUDA' in path:
            print("CUDA found in PATH, but nvcc failed.")

    print('--- 3. 框架层 (torch.__version__) ---')
    try:
        print(torch.__version__)
    except:
        print("Torch import failed?")

    print('--- 4. 编译层 (torch.version.cuda) ---')
    try:
        print(torch.version.cuda)
    except:
        print("torch.version.cuda not available")

    print('--- 5. 链接状态 (torch.cuda.is_available) ---')
    try:
        print(f'CUDA Available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'Device Name: {torch.cuda.get_device_name(0)}')
    except Exception as e:
        print(f"CUDA check crashed: {e}")

if __name__ == "__main__":
    run_diag()
