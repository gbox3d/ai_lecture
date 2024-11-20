#%%
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# 모델 정의 (다층 퍼셉트론 사용)
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.layer1 = nn.Linear(2, 4)  # 입력층에서 은닉층으로 (2 -> 4)
        self.layer2 = nn.Linear(4, 1)  # 은닉층에서 출력층으로 (4 -> 1)
        self.activation = nn.Sigmoid()  # 활성화 함수

    def forward(self, x):
        x = self.activation(self.layer1(x))  # 은닉층 계산
        x = self.activation(self.layer2(x))  # 출력층 계산
        return x