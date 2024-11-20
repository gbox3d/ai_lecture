#%%
import torch
from torch.utils.data import Dataset, DataLoader

class MyXorDataset(Dataset):
    def __init__(self,X,Y):
        # 데이터셋 초기화
        self.x_data = X
        self.y_data = Y

    def __len__(self):
        # 데이터셋의 크기 반환
        return len(self.x_data)

    def __getitem__(self, idx):
        # 인덱스에 해당하는 데이터 반환
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y
# %%
