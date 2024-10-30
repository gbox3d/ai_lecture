#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#%% gpu 사용하기위하여 디바이스 얻기 
# device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device : {device} ")
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


# 금액 열에서 '원'을 제거하고 숫자 타입으로 변환
price_columns = ['win1_pric', 'win2_pric', 'win3_pric', 'win4_pric', 'win5_pric']
for col in price_columns:
    data[col] = data[col].str.replace('원', '').str.replace(',', '').astype(np.int64)

data.info()



#%% 'iso'를 인덱스로 설정하고 정렬
data.set_index('iso', inplace=True)
data.sort_index(inplace=True)

# 결과 확인
data.head()
# %%

# 필요한 열만 사용 (iso 제외)
lotto_numbers = data[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'cb']]
# %%
lotto_numbers.head()
# %%
# 데이터 정규화 (0~1 범위로 스케일링)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(lotto_numbers)
# %%
print(scaled_data.shape)
# %%
# 시퀀스 생성 함수
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# 시퀀스 길이 설정
sequence_length = 10

# 시퀀스 생성
X, y = create_sequences(scaled_data, sequence_length)
# %%
print(X.shape)
print(y.shape)
# %%

# 데이터셋을 훈련 및 테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 텐서로 변환
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

#%%
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# %%

# LSTM 모델 정의
class LottoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LottoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 마지막 LSTM의 출력만 사용
        return out

# 모델 인스턴스 생성
input_size = 7  # c1~cb의 총 7개 입력
hidden_size = 64  # LSTM의 은닉 상태 크기
output_size = 7  # 예측해야 할 c1~cb의 7개 출력
num_layers = 2  # LSTM 레이어 수

model = LottoLSTM(input_size, hidden_size, output_size, num_layers).to(device)

# %%

# 손실 함수 및 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 훈련
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# %%
# 모델 평가
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')
# %%
# 예측된 결과를 스케일 원래 범위로 복원
predicted = scaler.inverse_transform(test_outputs.detach().numpy())
print("Predicted Lotto Numbers (Next Draw):", predicted[-1])
# %%
X_tensor = torch.tensor(X, dtype=torch.float32)
last_sequence = X_tensor[-1].unsqueeze(0)  # 마지막 시퀀스
# %%
print(last_sequence)

# %%
with torch.no_grad():
    next_draw_prediction = model(last_sequence)
# %%

# 예측된 결과를 원래 스케일로 역변환
predicted_numbers = scaler.inverse_transform(next_draw_prediction.detach().numpy())

# 예측된 다음 회차 번호 출력
print("Predicted Lotto Numbers (Next Draw):", predicted_numbers[0])
# %%

# 예측된 다음 회차 번호 출력

# 예측된 숫자를 반올림하여 정수로 변환
predicted_numbers_rounded = np.round(predicted_numbers).astype(int)

print("Predicted Lotto Numbers (Next Draw):", predicted_numbers_rounded[0])
# %%
