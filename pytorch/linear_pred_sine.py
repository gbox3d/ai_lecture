#%%
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import math
import matplotlib.pyplot as plt

print(f'torch version {torch.__version__}')

#%%
x = torch.linspace(0, 2*math.pi, 1000).unsqueeze(1) # 1000 x 1
y = torch.sin(x)



#%%

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='sin(x)', color='blue')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Function')
plt.legend()
plt.show()


# %% 모델 정의
class SineApproximator(nn.Module):
    def __init__(self):
        super(SineApproximator, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 모델 인스턴스 생성
model = SineApproximator()


# 하이퍼파라미터 및 옵티마이저 설정
learning_rate = 0.01
epochs = 5000
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()


#%%# 모델 훈련
loss_values = []  # Loss 값을 저장할 리스트

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    
    # Loss 값을 리스트에 저장
    loss_values.append(loss.item())

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')


#%% Loss 값 시각화
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), loss_values, label='Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()
        

#%%
# 결과 시각화
x_test = torch.linspace(0, 2 * math.pi, 1000).unsqueeze(1)
y_test = model(x_test)

plt.figure(figsize=(10, 5))
plt.plot(x_test, y_test.detach().numpy(), label='Prediction', color='blue')
plt.plot(x_test, torch.sin(x_test), label='Ground Truth', color='red')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Function Approximation')
plt.legend()
plt.show()
