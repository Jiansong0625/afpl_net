import torch
from dla34 import DLAWrapper

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = DLAWrapper(pretrained=False)
    print(model)  # 打印模型结构
    out = model(x)
    for i, o in enumerate(out):
        print(f"Output {i} shape:", o.shape)