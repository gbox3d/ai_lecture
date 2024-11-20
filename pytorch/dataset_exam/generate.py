#%%
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

print('module load complete')
print(f'torch version {torch.__version__}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device : {device}')

from my_dataset import MyXorDataset

#%%

X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 데이터셋 생성
dataset = MyXorDataset(X,Y)

#%%
torch.save(dataset, 'xor_dataset.pth')
 

#%%
# DataLoader 생성
# batch_size를 1로 설정하여 1개의 데이터만 반환하도록 설정
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 데이터 로딩 및 출력 예시
for batch_idx, (inputs, targets) in enumerate(dataloader):
    print(f"Batch {batch_idx+1}:")
    print(f"Inputs:\n{inputs}")
    print(f"Targets:\n{targets}\n")

# %% batch size 2
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
for batch_idx, (inputs, targets) in enumerate(dataloader):
    print(f"Batch {batch_idx+1}:")
    print(f"Inputs:\n{inputs}")
    print(f"Targets:\n{targets}\n")
# %%
