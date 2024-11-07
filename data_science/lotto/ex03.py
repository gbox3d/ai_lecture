#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import random

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#%%

# 디바이스 설정 (GPU 사용 가능 시 GPU, 아니면 CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")

#%%
# CSV 파일 읽기
data_1 = pd.read_csv('../datasets/lotto_1.csv')
data_2 = pd.read_csv('../datasets/lotto_2.csv')

# 두 데이터 병합
data = pd.concat([data_1, data_2], ignore_index=True)

# 병합된 데이터 미리보기
# data_combined.head()

# CSV 파일 읽기
# data = pd.read_csv('lotto_1.csv')


# # 금액 열에서 '원'을 제거하고 숫자 타입으로 변환
# price_columns = ['win1_pric', 'win2_pric', 'win3_pric', 'win4_pric', 'win5_pric']
# for col in price_columns:
#     data[col] = data[col].str.replace('원', '').str.replace(',', '').astype(np.int64)

#%%
data.info()
data.head()

#%% 'iso'를 인덱스로 설정하고 정렬
data.set_index('iso', inplace=True)
data.sort_index(inplace=True)

# 결과 확인
data.head()

#%%
# 필요한 열만 사용 (iso 제외)
lotto_numbers = data[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'cb']]
lotto_numbers.head()
# %%

# 당첨번호를 원핫 인코딩 벡터로 변환하는 함수
def numbers2ohbin(numbers):
    ohbin = np.zeros(45)  # 45개의 빈 칸 생성
    for num in numbers:
        ohbin[int(num)-1] = 1
    return ohbin

# 원핫 인코딩 벡터를 번호로 변환하는 함수
def ohbin2numbers(ohbin):
    numbers = [i+1 for i in range(len(ohbin)) if ohbin[i] == 1.0]
    return numbers

# 데이터 로드 (이미 데이터프레임 'data'가 있다고 가정)
# 로또 번호만 추출
lotto_numbers = data[['c1', 'c2', 'c3', 'c4', 'c5', 'c6']].values

# 원핫 인코딩 적용
ohbins = list(map(numbers2ohbin, lotto_numbers))

# 입력(X)과 출력(y) 생성
x_samples = ohbins[:-1]
y_samples = ohbins[1:]

# 샘플 확인
print("X[0]:", x_samples[0])
print("Y[0]:", y_samples[0])

#%%

#원핫인코딩으로 표시
print("ohbins")
print("X[0]: " + str(x_samples[0]))
print("Y[0]: " + str(y_samples[0]))

#번호로 표시
print("numbers")
print("X[0]: " + str(ohbin2numbers(x_samples[0])))
print("Y[0]: " + str(ohbin2numbers(y_samples[0])))

#%%
print("X[1]: " + str(ohbin2numbers(x_samples[1])))
print("Y[1]: " + str(ohbin2numbers(y_samples[1])))


# %% 데이터셋 분할
# 전체 데이터 개수
total_samples = len(x_samples)
print("total samples:", total_samples)

# 인덱스 설정
train_idx = (0, int(total_samples * 0.8))
val_idx = (int(total_samples * 0.8), int(total_samples * 0.9))
test_idx = (int(total_samples * 0.9), total_samples)

print("train: {0}, val: {1}, test: {2}".format(train_idx, val_idx, test_idx))


# %%
class LottoDataset(Dataset):
    def __init__(self, x_samples, y_samples, idx_range):
        self.x_samples = x_samples[idx_range[0]:idx_range[1]]
        self.y_samples = y_samples[idx_range[0]:idx_range[1]]
        
    def __len__(self):
        return len(self.x_samples)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.x_samples[idx], dtype=torch.float32)
        y = torch.tensor(self.y_samples[idx], dtype=torch.float32)
        return x, y

# 데이터셋 생성
train_dataset = LottoDataset(x_samples, y_samples, train_idx)
val_dataset = LottoDataset(x_samples, y_samples, val_idx)
test_dataset = LottoDataset(x_samples, y_samples, test_idx)

# 데이터로더 생성
batch_size = 64  # 혹은 GPU 메모리에 맞게 128, 256 등으로 조절

train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# %% 모델정의

class LottoPredictor(nn.Module):
    def __init__(self):
        super(LottoPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=45, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 45)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        x = x.unsqueeze(1)  # (batch, seq_len=1, input_size)
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]  # 마지막 시퀀스 출력
        out = self.fc(out)
        out = self.sigmoid(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # (num_layers, batch_size, hidden_size)
        return (torch.zeros(1, batch_size, 128).to(device),
                torch.zeros(1, batch_size, 128).to(device))


# %%
model = LottoPredictor().to(device) # 모델 생성

criterion = nn.BCELoss() # 손실함수
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 옵티마이저

#%%
print(next(model.parameters()).device)

#%%
x_batch, y_batch = next(iter(train_loader))
print(x_batch.device)
print(y_batch.device)

#%%
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    train_accuracies = []

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        batch_size = x_batch.size(0)
        hidden = model.init_hidden(batch_size)

        optimizer.zero_grad()
        output, hidden = model(x_batch, hidden)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        # hidden state detach
        hidden = (hidden[0].detach(), hidden[1].detach())

        train_losses.append(loss.item())

        # 정확도 계산
        preds = (output >= 0.5).float()
        correct = (preds == y_batch).float().mean()
        train_accuracies.append(correct.item())

    # 검증 손실 및 정확도 계산
    model.eval()
    val_losses = []
    val_accuracies = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            batch_size = x_batch.size(0)
            hidden = model.init_hidden(batch_size)

            output, hidden = model(x_batch, hidden)
            loss = criterion(output, y_batch)
            val_losses.append(loss.item())

            preds = (output >= 0.5).float()
            correct = (preds == y_batch).float().mean()
            val_accuracies.append(correct.item())

            hidden = (hidden[0].detach(), hidden[1].detach())

    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
          .format(epoch+1, num_epochs, np.mean(train_losses), np.mean(train_accuracies),
                  np.mean(val_losses), np.mean(val_accuracies)))


# %%

# 예측하고자 하는 특정 회차 인덱스를 지정
# 예를 들어, 마지막 회차라면 -1로 지정하고, 특정 회차 번호라면 해당 인덱스를 넣으면 됩니다.
specific_round_index = 600  # 마지막 회차, 또는 원하는 회차의 인덱스로 변경
specific_round_numbers = lotto_numbers[specific_round_index]

# 선택한 회차 번호를 원핫 인코딩으로 변환
specific_round_input = torch.tensor(numbers2ohbin(specific_round_numbers), dtype=torch.float32).to(device)
specific_round_input = specific_round_input.unsqueeze(0)  # 배치 차원을 추가

# 모델의 히든 상태 초기화
batch_size = 1  # 단일 회차 예측이므로 배치 사이즈는 1
hidden = model.init_hidden(batch_size)

# 예측 모드로 모델 실행
model.eval()
with torch.no_grad():
    # 모델에 입력하여 예측
    output, hidden = model(specific_round_input, hidden)

    # 예측된 원핫 벡터를 로또 번호로 변환
    predicted_ohbin = (output >= 0.5).float().cpu().numpy()
    predicted_numbers = ohbin2numbers(predicted_ohbin[0])

print(f"{specific_round_index}번째 회차에 대한 예측 번호:", predicted_numbers)

# %%
# 'iso'를 인덱스로 사용한다고 가정하고, 600회차 당첨 번호 출력하기
specific_round = 600

# 특정 회차의 데이터 선택
specific_round_data = data.loc[specific_round]

# 당첨 번호와 보너스 번호 출력
winning_numbers = specific_round_data[['c1', 'c2', 'c3', 'c4', 'c5', 'c6']].values
bonus_number = specific_round_data['cb']

print(f"{specific_round}회차 당첨 번호:", winning_numbers)
print(f"{specific_round}회차 보너스 번호:", bonus_number)

# %%
