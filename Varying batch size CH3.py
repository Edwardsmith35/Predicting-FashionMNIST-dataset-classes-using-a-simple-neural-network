from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
device = "cuda" if torch.cuda.is_available() else "cpu"
from torchvision import datasets
data_folder = '~/data/FMNIST' # This can be any directory you want to 
# download FMNIST to
fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
tr_images = fmnist.data # shape (N, 28, 28) Where N is the Number of images
tr_targets = fmnist.targets # shape (N,)

val_fmnist =datasets.FashionMNIST(data_folder,download=True, train=False)
val_images = val_fmnist.data
val_targets = val_fmnist.targets

class FMNIST_DATASET(Dataset):
    def __init__(self,x , y):
      x = x.float() /255
      self.x = x.view(-1, 28*28) # -1 becomes N Number of images, or use x.flatten(1)
      self.y = y.long()
    
    def __len__(self):
      return len(self.x)
    
    def __getitem__(self, ix):
      return self.x[ix].to(device) , self.y[ix].to(device)

def get_model(input_dim=784, lr=1e-2):
    model = nn.Sequential(
                nn.Linear(input_dim, 1000),
                nn.ReLU(),
                nn.Linear(1000, 10)
            ).to(device)
    loss_func = nn.CrossEntropyLoss()
    from torch.optim import SGD, Adam
    optimizer = SGD(model.parameters(), lr)
    return model, optimizer, loss_func

# Corrected get_data
def get_data(batch_size=32):
    train_DS = FMNIST_DATASET(tr_images, tr_targets)
    train_DL = DataLoader(train_DS, batch_size=batch_size, shuffle=True)
    val_DS = FMNIST_DATASET(val_images, val_targets)
    val_DL = DataLoader(val_DS, batch_size=len(val_images), shuffle=False) # Use val_DS
    return train_DL, val_DL

def train_batch(x, y, model, loss_func, optimizer):
    model.train() # set it to train mode
    prediction = model(x)
    batch_loss = loss_func(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(model, x, y):
    model.eval() # set model to evaluation mode
    prediction = model(x)
    max_values, argmaxes = prediction.max(-1) # get the maximumum value from the last dimension
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def val_loss(model, x, y):
    prediction = model(x)
    val_loss = loss_func(prediction, y)
    return val_loss.item()

train_DL, val_DL = get_data()
model, optimizer, loss_func = get_model()   
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
for epoch in range(5):
    print(f"Training Epoch: {epoch}")
    train_epoch_losses, train_epoch_accuracies = [], []
    for ix, batch in enumerate(iter(train_DL)):
        x, y = batch
        avg_batch_loss = train_batch(x, y, model, loss_func, optimizer)
        train_epoch_losses.append(avg_batch_loss) 
    train_epoch_loss = np.array(train_epoch_losses).mean() 
    
    for ix, batch in enumerate(iter(train_DL)):
        x, y = batch
        is_correct = accuracy(model, x, y)
        train_epoch_accuracies.extend(is_correct)
    train_epoch_accuracy = np.mean(train_epoch_accuracies)
    
    for ix, batch in enumerate(iter(val_DL)):
        x, y = batch
        val_is_correct = accuracy(model, x, y)
        validation_loss = val_loss(model, x, y)
  
    val_epoch_accuracy = np.mean(val_is_correct)
    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_losses.append(validation_loss)
    val_accuracies.append(val_epoch_accuracy)

epochs = np.arange(5)+1
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
%matplotlib inline
plt.subplot(211)
plt.plot(epochs, train_losses, 'bo', label='Training loss')
plt.plot(epochs, val_losses, 'r', label='Validation loss')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation loss when batch size is 32')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
plt.show()
plt.subplot(212)
plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation accuracy when batch size is 32')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
plt.legend()
plt.grid('off')
plt.show()
      
    
_, Data = get_data(2)
for ix, batch in enumerate(iter(Data)):
    x, y = batch
    with torch.no_grad():
        prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    
    classes = fmnist.classes
    # The original code used 'Indices' which contains indices for all data points.
    # We should iterate through each index in 'argmaxes' instead.
    for index in argmaxes:  
        predicted_class = classes[index.item()]  # Get the predicted class for each data point.
        # Assuming you want to display the prediction for the first data point
        plt.imshow(x[0].reshape(28, 28))  # Reshape and display the image.
        plt.title(f"{predicted_class=}")  # Set the title with the predicted class.
        break # Break the inner loop after processing the first data point
    break # Break the outer loop after processing the first batch

classes = fmnist.classes
import random
rand = int(random.random()*1000)
image = val_images[rand,:,:].float()  # shape (1, 28, 28)
with torch.no_grad():
    prediction = model(image.reshape(28*28))
print(prediction)
max_values, indexes = prediction.max(-1) # returns a tuple of (max_values, indexs)
max_index = prediction.argmax() # or use this to get the index of max value
print(f"{max_index=}")
print(f"{max_values=}")
print(f"{indexes=}")
predicted_class = classes[indexes]
plt.imshow(image.reshape(28, 28), cmap="gray")  # Reshape and display the image.
plt.title(f"{predicted_class=}")
plt.show()