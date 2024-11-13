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



#%%
# CSV 파일 읽기
data_1 = pd.read_csv('../../datasets/lotto_1.csv')
data_2 = pd.read_csv('../../datasets/lotto_2.csv')

# 두 데이터 병합
data = pd.concat([data_1, data_2], ignore_index=True)

#%%
# 필요한 열만 사용 (iso 제외)
lotto_numbers = data[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'cb']]
lotto_numbers.head()
# %%

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


# 데이터셋 생성
train_dataset = LottoDataset(x_samples, y_samples, train_idx)
val_dataset = LottoDataset(x_samples, y_samples, val_idx)
test_dataset = LottoDataset(x_samples, y_samples, test_idx)

# 데이터로더 생성
batch_size = 64  # 혹은 GPU 메모리에 맞게 128, 256 등으로 조절

train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)


def getloaders():
    return train_loader, val_loader, test_loader