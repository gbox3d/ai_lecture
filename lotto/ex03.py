#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#%%


# CSV 파일 읽기
data_1 = pd.read_csv('lotto_1.csv')
data_2 = pd.read_csv('lotto_2.csv')

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

#%%
data.info()
data.head()

#%% 'iso'를 인덱스로 설정하고 정렬
data.set_index('iso', inplace=True)
data.sort_index(inplace=True)

# 결과 확인
data.head()
# %%

#%%
# 필요한 열만 사용 (iso 제외)
lotto_numbers = data[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'cb']]
lotto_numbers.head()
# %%
