#%%
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

from torchvision.datasets.cifar import CIFAR10

transform = T.Compose( [
    T.RandomCrop((32,32), padding=4),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=(0.4914,0.4822,0.4465), std=(0.247,0.243,0.261))
])

test_data = CIFAR10(root="./datasets", train=False, download=True, transform=transform)
#%%

for i in range(10):
    plt.subplot(2, 5, i+1)
    img_tensor = test_data[i][0]
    img_tensor = torch.clamp(img_tensor, 0, 1) 
    plt.imshow(img_tensor.permute(1, 2, 0))
    
plt.show()

# %%

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(test_data.data[i])
plt.show()
# %%
