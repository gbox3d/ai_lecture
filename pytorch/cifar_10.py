#%%
import matplotlib.pyplot as plt
import torchvision.transforms as T

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor



#%% cifar10 데이터셋 다운로드
training_data = CIFAR10(root="./datasets", train=True, download=True, transform=ToTensor())
test_data = CIFAR10(root="./datasets", train=False, download=True, transform=ToTensor())

# %%
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(training_data.data[i])
    print(training_data.data[i].shape)
plt.show()
    

# %%
print(training_data.data.shape)

# %%
