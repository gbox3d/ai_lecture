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
# x_samples = ohbins[:-1]
# y_samples = ohbins[1:]

sequence_length = 5  # 예시로 시퀀스 길이를 5로 설정
x_samples = [ohbins[i:i+sequence_length] for i in range(len(ohbins) - sequence_length)]
y_samples = ohbins[sequence_length:]


# 샘플 확인
print("X[0]:", x_samples[0])
print("Y[0]:", y_samples[0])


#%%

#원핫인코딩으로 표시
print("ohbins")
print("X[0]: " + str(x_samples[0][0]))
print("Y[0]: " + str(y_samples[0]))

#%%
#번호로 표시
print("numbers")
print("X[0]: " + str(ohbin2numbers(x_samples[0][0])))
print("Y[0]: " + str(ohbin2numbers(y_samples[0])))



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
        # x의 형태는 (batch_size, seq_len, input_size)
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
# 공통 함수 정의
def calculate_correct_numbers(output, y_batch, k):
    # 상위 k개의 번호 선택
    _, topk_indices = torch.topk(output, k, dim=1)
    preds = torch.zeros_like(output)
    preds.scatter_(1, topk_indices, 1)

    # 각 샘플에서 맞춘 번호 개수 계산
    num_correct_per_sample = (preds * y_batch).sum(dim=1)
    return num_correct_per_sample.cpu().numpy(), preds


num_epochs = 300

k = 6  # 로또 번호는 매 회차 6개이므로, 상위 6개 번호를 선택합니다.

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    train_correct_numbers = []

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

        # 정확도 계산: 각 샘플별로 맞춘 번호 개수 계산
        num_correct_per_sample, _ = calculate_correct_numbers(output, y_batch, k)
        train_correct_numbers.extend(num_correct_per_sample)

    avg_correct_numbers = np.mean(train_correct_numbers)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(train_losses):.4f}, Correct Numbers: {avg_correct_numbers:.2f}")

# %% 검증 및 테스트 데이터셋에서 모델 평가

model.eval()  # 모델을 평가 모드로 전환
with torch.no_grad():
    # 검증 데이터셋 평가
    val_losses = []
    val_correct_numbers = []

    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        batch_size = x_batch.size(0)
        hidden = model.init_hidden(batch_size)

        # 모델 예측
        output, hidden = model(x_batch, hidden)
        loss = criterion(output, y_batch)

        val_losses.append(loss.item())

        # 정확도 계산: 각 샘플별로 맞춘 번호 개수 계산
        num_correct_per_sample, _ = calculate_correct_numbers(output, y_batch, k)
        val_correct_numbers.extend(num_correct_per_sample)

    avg_val_loss = np.mean(val_losses)
    avg_val_correct_numbers = np.mean(val_correct_numbers)

    print(f"Validation Loss: {avg_val_loss:.4f}, Avg Correct Numbers: {avg_val_correct_numbers:.2f}")

    # 테스트 데이터셋 평가
    test_losses = []
    test_correct_numbers = []

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        batch_size = x_batch.size(0)
        hidden = model.init_hidden(batch_size)

        # 모델 예측
        output, hidden = model(x_batch, hidden)
        loss = criterion(output, y_batch)

        test_losses.append(loss.item())

        # 정확도 계산: 각 샘플별로 맞춘 번호 개수 계산
        num_correct_per_sample, _ = calculate_correct_numbers(output, y_batch, k)
        test_correct_numbers.extend(num_correct_per_sample)

    avg_test_loss = np.mean(test_losses)
    avg_test_correct_numbers = np.mean(test_correct_numbers)

    print(f"Test Loss: {avg_test_loss:.4f}, Avg Correct Numbers: {avg_test_correct_numbers:.2f}")


#%%
# %% 특정 회차 예측 및 실제와 비교
# %% 특정 회차 예측 및 실제와 비교
sequence_length = 5  # 시퀀스 길이
specific_round_index = 100  # 예측하려는 회차 인덱스

# y_samples에서의 인덱스 계산
y_index = specific_round_index - sequence_length

# 입력 시퀀스 준비 (훈련 데이터와 동일한 방식)
input_sequence = x_samples[y_index]
input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)  # 배치 차원 추가

# 모델의 히든 상태 초기화
batch_size = 1
hidden = model.init_hidden(batch_size)

# 모델 예측
model.eval()  # 모델을 평가 모드로 전환
with torch.no_grad():
    output, hidden = model(input_sequence, hidden)

    # 상위 k개의 번호 선택
    _, topk_indices = torch.topk(output, k=k, dim=1)
    preds = torch.zeros_like(output)
    preds.scatter_(1, topk_indices, 1)

    # 예측된 번호 추출
    predicted_ohbin = preds.cpu().numpy()
    predicted_numbers = ohbin2numbers(predicted_ohbin[0])

print(f"{specific_round_index}번째 회차에 대한 예측 번호:", predicted_numbers)

# 특정 회차의 실제 당첨 번호 출력
specific_round_data = data.iloc[specific_round_index]
winning_numbers = specific_round_data[['c1', 'c2', 'c3', 'c4', 'c5', 'c6']].values.astype(int)
bonus_number = int(specific_round_data['cb'])

print(f"{specific_round_index}회차 당첨 번호:", winning_numbers)
print(f"{specific_round_index}회차 보너스 번호:", bonus_number)

# %%
