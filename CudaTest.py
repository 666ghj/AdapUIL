"""
@File    ：CudaTest.py
@Author  : silvan
@Time    : 2024/3/11 21:58

"""
import torch

print(torch.__version__)#查看torch版本

print(torch.cuda.is_available())#看安装好的torch和cuda能不能用，也就是看GPU能不能用

print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda())
