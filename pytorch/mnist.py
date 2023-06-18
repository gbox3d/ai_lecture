#%%
import matplotlib.pyplot as plt

from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

import torch
import torch.nn as nn
from torch.optim.adam import Adam


#%% prepare data

training_data = MNIST(root='./datasets', train=True, download=True, transform=ToTensor())
test_data = MNIST(root='./datasets', train=False, download=True, transform=ToTensor())

print(f'training data size: {len(training_data)}')
print(f'test data size: {len(test_data)}')

#%%
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(training_data[i][0].squeeze(), cmap='gray')
    plt.title(training_data[i][1])
    plt.axis('off')

# %%
from torch.utils.data import DataLoader

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

#%%

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'using {device}')

model = nn.Sequential(
    nn.Linear(784, 128),  # 28*28 = 784
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
model.to(device) # move model to device

lr = 1e-3
optim = Adam(model.parameters(), lr=lr)

for epch in range(20) :
    for data,label in train_loader :
        optim.zero_grad()
        data = torch.reshape(data, (-1, 784)).to(device)
        preds = model(data)
        
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()
    print(f'epoch {epch} loss {loss.item()}')
    
    
# %% 저장 

torch.save(model.state_dict(), 'mnist.pth')
# %%

num_correct = 0

with torch.no_grad() :
    for data, label in test_loader :
        data = torch.reshape(data, (-1, 784)).to(device)
        preds = model(data.to(device=device))
        num_correct += torch.sum(torch.argmax(preds, dim=1) == label.to(device)).item()
print(f'accuracy : {num_correct/len(test_data)}')
# %%

print(test_loader)

# %%
# 테스트 데이터셋의 첫 번째 이미지와 레이블 가져오기
first_test_image, first_test_label = test_data[0]

# 이미지 시각화
plt.imshow(first_test_image.squeeze(), cmap='gray')
plt.title(f'Ground Truth: {first_test_label}')
plt.axis('off')
plt.show()

#%%
# 모델 추론
with torch.no_grad():
    input_image = first_test_image.view(-1, 784).to(device)  # 이미지를 (1, 784) 형태로 변환하고 장치로 이동
    output = model(input_image)  # 모델의 예측값 계산
    print(output.data)
    print(output.data.max(1))
    prediction = torch.argmax(output, dim=1).item()  # 가장 높은 확률을 가진 숫자를 가져옴

print(f'Prediction: {prediction}')

# %%
