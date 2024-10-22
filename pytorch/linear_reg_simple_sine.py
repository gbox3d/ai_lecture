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
class SimpleSineModel(nn.Module):
    def __init__(self):
        super(SimpleSineModel, self).__init__()
        # Sequential 모델과 동일한 계층 구성
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

simple_model = SimpleSineModel()

#%%
learning_rate = 0.01
epochs = 100000
optimizer = optim.Adam(simple_model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

loss_values = []  # Loss 값을 저장할 리스트
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = simple_model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    
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
y_test = simple_model(x_test)

plt.figure(figsize=(10, 5))
plt.plot(x_test, y_test.detach().numpy(), label='Prediction', color='blue')
plt.plot(x_test, torch.sin(x_test), label='Ground Truth', color='red')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Function Approximation')
plt.legend()
plt.show()
# %%

