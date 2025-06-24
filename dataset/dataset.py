from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from config import device

class FMNIST_DATASET(Dataset):
    def __init__(self,x , y):
      x = x.float() /255
      self.x = x.view(-1, 28*28) # -1 lets pytorch infer the correct Number of images
      self.y = y.long()
    
    def __len__(self):
      return len(self.x)
    
    def __getitem__(self, ix):
      return self.x[ix].to(device) , self.y[ix].to(device)

def prepare_FMNIST_train_data(data_folder = '~/data/FMNIST'):
    fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
    tr_images = fmnist.data # shape (N, 28, 28) Where N is the Number of images
    tr_targets = fmnist.targets # shape (N,)
    return tr_images, tr_targets, fmnist.classes

def prepare_FMNIST_validation_data(data_folder = '~/data/FMNIST'):
    val_fmnist = datasets.FashionMNIST(data_folder,download=True, train=False)
    val_images = val_fmnist.data
    val_targets = val_fmnist.targets
    return val_images, val_targets, val_fmnist.classes



# Corrected get_data
def train_val_DL(tr_images, tr_targets, val_images, val_targets, batch_size=32):
    train_DS = FMNIST_DATASET(tr_images, tr_targets)
    train_DL = DataLoader(train_DS, batch_size=batch_size, shuffle=True)
    val_DS = FMNIST_DATASET(val_images, val_targets)
    val_DL = DataLoader(val_DS, batch_size=len(val_images), shuffle=False) # Use val_DS
    return train_DL, val_DL
