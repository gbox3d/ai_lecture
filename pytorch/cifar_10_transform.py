#%%
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

from torchvision.datasets.cifar import CIFAR10
#%%
_transform = T.Compose([
    # T.ToPILImage(),
    T.RandomCrop((32,32), padding=4),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor()
])

cmp_training_data = CIFAR10(root="./datasets", train=True, download=True, transform=_transform)
cmp_test_data = CIFAR10(root="./datasets", train=False, download=True, transform=_transform)
#%% 변형본 출력 
print(cmp_training_data[0][0].shape)
for i in range(10):
    plt.subplot(2, 5, i+1)
    # plt.imshow(cmp_training_data[i].permute(1, 2, 0))
    plt.imshow(cmp_training_data[i][0].permute(1, 2, 0))
    # print(cmp_training_data[i][1])
    
plt.show()

# %% 원본 출력
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(cmp_training_data.data[i])
plt.show()

