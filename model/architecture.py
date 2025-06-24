import torch.nn as nn
from config import device

def get_model(input_dim=784, lr=1e-2):
    model = nn.Sequential(
                nn.Linear(input_dim, 1000),
                nn.ReLU(),
                nn.Linear(1000, 10)
            ).to(device)
    loss_func = nn.CrossEntropyLoss()
    from torch.optim import SGD #, Adam
    optimizer = SGD(model.parameters(), lr)
    return model, optimizer, loss_func
