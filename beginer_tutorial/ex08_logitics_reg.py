#%%
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

#%%
# 0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#%%
print(X_train.shape)
print(y_train.shape)
print(X_train[0])
#%%
# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert to torch tensors
X_train_torch = torch.from_numpy(X_train.astype(np.float32))
X_test_torch = torch.from_numpy(X_test.astype(np.float32))
y_train_torch = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
y_test_torch = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)


#%% 그래프로 확인
# Feature names
feature_names = bc.feature_names

# Plot using the original scaled numpy arrays (not the torch tensors)
plt.figure(figsize=(10, 6))
for class_value in np.unique(y_test):
    row_ix = np.where(y_test == class_value)
    plt.scatter(X_test[row_ix, 0], X_test[row_ix, 1], label=f'Class {class_value} ({bc.target_names[class_value]})')

plt.title('Breast Cancer Test Set Features Visualization')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend()
plt.show()


#%%

# 1) Model
# Linear model f = wx + b , sigmoid at the end
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_features)

# 2) Loss and optimizer
num_epochs = 1000
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#%%
# 3) Training loop
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(X_train_torch)
    loss = criterion(y_pred, y_train_torch)

    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#%%
with torch.no_grad():
    y_predicted = model(X_test_torch)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test_torch).sum() / float(y_test_torch.shape[0])
    print(f'accuracy: {acc.item():.4f}')
# %%

# Create a figure
plt.figure(figsize=(12, 6))

# Plot actual classes
plt.subplot(1, 2, 1)
for class_value in np.unique(y_test_torch):
    row_ix = np.where(y_test_torch == class_value)
    plt.scatter(X_test_torch[row_ix, 0], X_test_torch[row_ix, 1], label=f'Actual Class {class_value}')

plt.title('Actual Test Set Labels')
plt.xlabel(bc.feature_names[0])
plt.ylabel(bc.feature_names[1])
plt.legend()

# Plot predicted classes
plt.subplot(1, 2, 2)
for class_value in np.unique(y_predicted_cls):
    row_ix = np.where(y_predicted_cls == class_value)
    plt.scatter(X_test_torch[row_ix, 0], X_test_torch[row_ix, 1], label=f'Predicted Class {class_value}')

plt.title('Predicted Test Set Labels')
plt.xlabel(bc.feature_names[0])
plt.ylabel(bc.feature_names[1])
plt.legend()

plt.tight_layout()
plt.show()
# %%
