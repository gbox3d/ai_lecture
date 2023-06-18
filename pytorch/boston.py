#%%
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.adam import Adam

print(f'torch version {torch.__version__}')


#%%
# Boston Housing Prices 데이터셋을 로드합니다.
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 데이터셋을 Bunch 객체로 변환합니다.
boston_dataset = {
    "data": data,
    "target": target,
    "feature_names": [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
    ],
    "DESCR": "Boston Housing Prices dataset",
}

# Bunch 객체를 반환합니다.
boston = fetch_openml(data_id=506, parser='auto')
boston.update(boston_dataset)

dataset = boston_dataset
print(dataset.keys())

# %% 판다스 데이터프레임으로 변환
dataFrame = pd.DataFrame(dataset['data'])
dataFrame.columns = dataset["feature_names"]
dataFrame["target"] = dataset["target"]

print(dataFrame.head()) # 상위 5개 데이터 출력

# %%
# data setup
X = dataFrame.iloc[:,:13].values # 13개의 feature
Y = dataFrame["target"].values # target

print( X[0] )
print(f'data size : {len(X)}')
# %%

# model 정의 
model = nn.Sequential(
    nn.Linear(13,100),
    nn.ReLU(),
    nn.Linear(100,1)
 )

# hyper parameter
batch_size = 100
learning_rate = 0.001
optim = Adam(model.parameters() , lr=learning_rate) # optimizer 정의

print(model)

#%% training
for epoch in range(2000) :
  for i in range( len(X) // batch_size ) :
    start = i*batch_size
    end = start + batch_size
    x = torch.FloatTensor(X[start:end])
    y = torch.FloatTensor(Y[start:end]).view(-1,1)

    optim.zero_grad()
    preds = model(x)
    loss = nn.MSELoss()(preds,y)
    loss.backward()
    optim.step()
  
  if epoch % 200 == 0 :
    print(f'epoch {epoch} loss : {loss.item()}')
# %% 검증

_data_index = 10
pred = model(torch.FloatTensor(X[_data_index,:13]))
ground_truth = Y[_data_index]
print (f'prediction : {pred} , ground truth : {ground_truth}')

#%%

# 예측값 계산
predictions = model(torch.FloatTensor(X)).detach().numpy().flatten()

# 데이터 인덱스
indices = np.arange(len(predictions))

# 라인 차트 그리기
plt.figure(figsize=(20, 5))
plt.plot(indices, predictions, label='Predictions', color='blue')
plt.plot(indices, Y, label='Ground Truth', color='red')
plt.xlabel('Data Index')
plt.ylabel('Housing Price')
plt.legend()
plt.show()


# %%
