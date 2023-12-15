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
   RandomCrop((32, 32), padding=4),  # ❶ 랜덤 크롭핑
   RandomHorizontalFlip(p=0.5),  # ❷ y축으로 뒤집기
   ToTensor(),  # ❸ 텐서로 변환
   # ❹ 이미지 정규화
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

#%%
# ❶ 학습 데이터와 평가 데이터 불러오기
training_data = CIFAR10(root="./datasets", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./datasets", train=False, download=True, transform=transforms)


# ❷ 데이터로더 정의
train_loader = DataLoader(training_data, batch_size=500, shuffle=True)
test_loader = DataLoader(test_data, batch_size=500, shuffle=False)


# ❸ 학습을 진행할 프로세서 설정
device = "cuda" if torch.cuda.is_available() else "cpu"


# ➍ CNN 모델 정의
model = CNN(num_classes=10)

# ➎ 모델을 device로 보냄
model.to(device)
#%%
# ❶ 학습률 정의
lr = 1e-3

# ❷ 최적화 기법 정의
optim = Adam(model.parameters(), lr=lr)

# 학습 루프 정의
for epoch in range(100):
    
    start_tick = time.time()
    # for data, label in train_loader:  # ➌ 데이터 호출
    for data, label in tqdm(train_loader, desc="Training Progress"):
        optim.zero_grad()  # ➍ 기울기 초기화

        preds = model(data.to(device))  # ➎ 모델의 예측

        # ➏ 오차역전파와 최적화
        loss = nn.CrossEntropyLoss()(preds, label.to(device)) 
        loss.backward() 
        optim.step() 
    
    print(f"epoch{epoch+1} time:{time.time()-start_tick}")
        

    if epoch==0 or epoch%10==9:  # 10번마다 손실 출력
        print(f"epoch{epoch+1} loss:{loss.item()}")


# 모델 저장
torch.save(model.state_dict(), "CIFAR.pth")
print("Saved PyTorch Model State to CIFAR.pth")
# %%

model.load_state_dict(torch.load("CIFAR.pth", map_location=device))

num_corr = 0

with torch.no_grad():
   for data, label in test_loader:

       output = model(data.to(device))
       preds = output.data.max(1)[1]
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")


# %%
