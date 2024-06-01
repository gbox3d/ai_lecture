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

# 8보다 큰수를 사용하면 더 정확한 예측을 할 수 있지만, 파라미터가 많아져서 학습이 느려짐
model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# 하이퍼파라미터 및 옵티마이저 설정
learning_rate = 0.01
epochs = 5000
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()


#%%# 모델 훈련
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
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