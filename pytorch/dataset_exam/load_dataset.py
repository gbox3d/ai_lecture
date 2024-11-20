#%%
import torch
from torch.utils.data import Dataset, DataLoader

print('module load complete')
print(f'torch version {torch.__version__}')

from my_dataset import MyXorDataset
#%%
data = torch.load('xor_dataset.pth')
print(data)

#%%
print(data.x_data)
print(data.y_data)

#%%
dataset = MyXorDataset(data.x_data, data.y_data)
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

