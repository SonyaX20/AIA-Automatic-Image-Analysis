## EX1
### Installation
```bash
pip install python-opencv
```
#### pytorch
first check the NVIDIA version in cmd.
![alt text](img/nvidia.png)
I install ```cuda-12.0.1```. <br>
Install pytorch with ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118``` <br>
Pytorch官网上会给出指定的安装指令：```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118``` <br>
测试用例：
```python
import torch

# 创建一个随机的张量
x = torch.rand(5, 3)
print("Random Tensor:")
print(x)

# 检查CUDA是否可用，并在可用时将张量移动到GPU
if torch.cuda.is_available():
    x = x.cuda()
    print("Tensor on CUDA:")
    print(x)
```
安装成功
![alt text](img/pytorch.png)