#%%
import torch
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize

from tqdm import tqdm

import time

from myVggNet import CNN


#%%
transforms = Compose([
#    RandomCrop((32, 32), padding=4),  # ❶ 랜덤 크롭핑
#    RandomHorizontalFlip(p=0.5),  # ❷ y축으로 뒤집기
   ToTensor(),  # ❸ 텐서로 변환
   # ❹ 이미지 정규화
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

#%%
# ❶ 학습 데이터와 평가 데이터 불러오기
test_data = CIFAR10(root="./datasets", train=False, download=True, 
                    transform=transforms
                    )


# ❷ 데이터로더 정의
test_loader = DataLoader(test_data, batch_size=500, shuffle=False)


# ❸ 학습을 진행할 프로세서 설정
device = "cuda" if torch.cuda.is_available() else "cpu"


# ➍ CNN 모델 정의
model = CNN(num_classes=10)

# ➎ 모델을 device로 보냄
model.to(device)
print("model loaded")
# %%

model.load_state_dict(torch.load("CIFAR.pth", map_location=device))

#%% test
num_corr = 0
with torch.no_grad():
   for data, label in test_loader:

       output = model(data.to(device))
       preds = output.data.max(1)[1]
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")


# %%
print(test_data.data.shape)

#%%

# test_data.data[0]를 PyTorch 텐서로 변환하고 전처리 적용
image = test_data.data[0]
image = transforms(image).unsqueeze(0)  # 차원 추가: [C, H, W] -> [1, C, H, W]

# 모델에 입력하고 예측
image = image.to(device)
with torch.no_grad():
    pred = model(image)
    print(pred)

# %%

# Ground truth 값 출력
ground_truth = test_data.targets[0]
print(f"Ground Truth: {ground_truth}")
# %%
