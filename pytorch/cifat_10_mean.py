#%%
import torch
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as T

#%%
training_data = CIFAR10(root="./datasets", train=True, download=True, transform=T.ToTensor())

#%%
imgs = [ item[0] for item in training_data ]

#%%
imgs = torch.stack(imgs,dim=0).numpy()

# %%
print(imgs.shape)
# %%
mean_r = imgs[:,0,:,:].mean()
mean_g = imgs[:,1,:,:].mean()
mean_b = imgs[:,2,:,:].mean()

print(mean_r, mean_g, mean_b)
# %%
#%% std
std_r = imgs[:,0,:,:].std()
std_g = imgs[:,1,:,:].std()
std_b = imgs[:,2,:,:].std()

print(std_r, std_g, std_b)
# %%
