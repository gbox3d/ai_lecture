
#%%
from datasetuitl import getloaders
# %%
train_loader, val_loader, test_loader = getloaders()
# %% dataloader test

print(train_loader.dataset)

# %%
for x_batch, y_batch in val_loader:
    print(y_batch)


# %%
