#%%
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


from model import XORModel

print('module load complete')
print(f'torch version {torch.__version__}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device : {device}')

#%%
dataset = torch.load('xor_dataset.pth')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# %%


# 모델 인스턴스 생성
model = XORModel()

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()  # 손실 함수: 평균 제곱 오차
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 옵티마이저: SGD

# 훈련 루프
num_epochs = 10000  # 에포크 수 설정
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # 순전파
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 일정 에포크마다 손실 출력
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#%% save model
torch.save(model, 'xor_model.pth')

# %%
