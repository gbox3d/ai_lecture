#%%
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# from model import XORModel

print('module load complete')
print(f'torch version {torch.__version__}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device : {device}')


#%% load model
model = torch.load('xor_model.pth')
print('finish load model')


#%%

# 모델 평가
with torch.no_grad():
    test_inputs = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
    test_outputs = model(test_inputs)
    print("\n예측 결과:")
    for input, output in zip(test_inputs, test_outputs):
        print(f"입력: {input.numpy()}, 출력: {output.item():.4f}")