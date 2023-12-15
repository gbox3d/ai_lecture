#%%
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
#%%
_transforms = T.Compose([
    T.ToPILImage(),
    T.RandomCrop((32,32), padding=4),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
   T.ToPILImage()
])

cmp_training_data = CIFAR10(root="./datasets", train=True, download=True, transform=_transforms)
cmp_test_data = CIFAR10(root="./datasets", train=False, download=True, transform=_transforms)

#%%
for i in range(10):
   plt.subplot(2, 5, i+1)
   plt.imshow(_transforms(cmp_training_data.data[i]))
plt.show()

#%% 변형본 출력 
print(cmp_training_data.data[0].shape)

# %% 원본 출력
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(cmp_training_data.data[i])
plt.show()


# %%
