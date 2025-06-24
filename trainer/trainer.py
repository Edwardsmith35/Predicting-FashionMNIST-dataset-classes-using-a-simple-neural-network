import numpy as np
from utils.utils import accuracy, val_loss


def train_batch(x, y, model, loss_func, optimizer):
    model.train()
    prediction = model(x)
    batch_loss = loss_func(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


def train_model(model, loss_func, optimizer, train_DL, val_DL, num_epochs):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    for epoch in range(num_epochs):
        print(f"Training Epoch: {epoch}")
        train_epoch_losses, train_epoch_accuracies = [], []
        for ix, batch in enumerate(iter(train_DL)):
            x, y = batch
            avg_batch_loss = train_batch(x, y, model, loss_func, optimizer)
            train_epoch_losses.append(avg_batch_loss)

            is_correct = accuracy(model, x, y)
            train_epoch_accuracies.extend(is_correct)

        train_epoch_accuracy = np.mean(train_epoch_accuracies)
        train_epoch_loss = np.array(train_epoch_losses).mean()

        for ix, batch in enumerate(iter(val_DL)):
            x, y = batch
            val_is_correct = accuracy(model, x, y)
            validation_loss = val_loss(model, x, y)

        val_epoch_accuracy = np.mean(val_is_correct)
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(validation_loss)
        val_accuracies.append(val_epoch_accuracy)

    return train_losses, val_losses, train_accuracies, val_accuracies
