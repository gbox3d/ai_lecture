#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

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
data.head()

#%%

# 당첨 번호 분포 시각화
plt.figure(figsize=(10, 6))
data[['c1', 'c2', 'c3', 'c4', 'c5', 'c6']].hist(bins=50)
plt.suptitle('Distribution of Winning Numbers', fontsize=16)

# 그래프 간격 조정
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# %% cb 의 분포
# 'cb' 열의 분포 시각화
plt.figure(figsize=(8, 6))
data['cb'].hist(bins=45, range=(1, 45))  # 1부터 45까지 범위 지정
plt.title('Distribution of CB', fontsize=12)
plt.xlabel('CB Numbers')
plt.ylabel('Frequency')

# x축 눈금을 1부터 45까지 설정하고 숫자 각도를 45도 기울임
plt.xticks(ticks=range(1, 46), rotation=-90)

# 그래프 출력
plt.show()
# %%
# 각 슬롯에서 가장 자주 나온 숫자 추출
columns = ['c1', 'c2', 'c3', 'c4', 'c5','c6']
recommendations = {}

for col in columns:
    most_common_number = data[col].value_counts().idxmax()  # 가장 많이 나온 숫자
    recommendations[col] = most_common_number
    

# %%
print(recommendations)
# %%

# 각 게임 슬롯(c1~c6)에 대한 가중치를 기반으로 확률적 번호 추천
def weighted_random_choice(column):
    value_counts = data[column].value_counts()  # 숫자 등장 빈도 계산
    numbers = value_counts.index.tolist()  # 등장한 숫자 리스트
    weights = value_counts.values.tolist()  # 각 숫자의 등장 빈도를 가중치로 사용
    
    # 가중치를 기반으로 숫자를 선택 (하나의 숫자만 선택)
    chosen_number = random.choices(numbers, weights=weights, k=1)
    return chosen_number[0]

#%% 추천 번호 생성
for col in columns:
    recommendations[col] = weighted_random_choice(col)

print("추천 번호:", recommendations)
# %%
