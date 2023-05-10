# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Download the Mnist dataset as a tensor
# %%
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.MNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

# %% [markdown]
# ## Prepare our data and make it iterable:
# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape (NCHW): {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

# %% [markdown]
# ## Making sure we are using the GPU:
# %%
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %% [markdown]
# ## Building and visualizing our CNN:
# %%
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pool_relu_stack = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(1, 1, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Flatten(),
            nn.Linear(400, 10),
            nn.Flatten(),
        )

    def forward(self, x):
        logits = self.conv_pool_relu_stack(x)
        # proba = nn.Softmax(1)(logits)
        return logits


model = CNN().to(device)

# %% [markdown]
# ### Testing without any training our model:
# %%
X = torch.rand(1, 28, 28, device=device)
proba = model(X)
print(f"Shape of the logits : {proba.shape}")

# %% [markdown]
# ### How big is our model and what is it's structure?
# %%
print(f"Model structure: {model}\n\n")

nbr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"We have {nbr_params} trainable parameters in our model")

# %% [markdown]
# ## Optimization hyperparameters:
# %%
learning_rate = 1e-3
batch_size = 128
epochs = 40
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# %% [markdown]
# ## Training loop
# %%
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, Y) in enumerate(dataloader):
        # Compute prediction and loss
        x = X.to(device=device)
        y = Y.to(device=device)
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, Y in dataloader:
            x = X.to(device=device)
            y = Y.to(device=device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# %% [markdown]
# ## Actual training:
# %%

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
