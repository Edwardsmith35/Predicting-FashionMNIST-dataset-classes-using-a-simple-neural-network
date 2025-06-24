from dataset.dataset import (
    prepare_FMNIST_train_data,
    prepare_FMNIST_validation_data,
    train_val_DL,
)
from config import data_folder, batch_size, num_epochs, input_dim, lr
from model.architecture import get_model
from trainer.trainer import train_model
from utils.test_model import test_model
from plotting.plotting import plot_train_val_loses


tr_images, tr_targets, classes = prepare_FMNIST_train_data(data_folder)

val_images, val_targets, val_classes = prepare_FMNIST_validation_data(data_folder)


train_DL, val_DL = train_val_DL(
    tr_images, tr_targets, val_images, val_targets, batch_size
)
model, optimizer, loss_func = get_model(input_dim, lr)

# train the model
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, loss_func, optimizer, train_DL, val_DL, num_epochs
)

# plot results
plot_train_val_loses(
    train_losses, val_losses, train_accuracies, val_accuracies, num_epochs
)

# Test the Model 2
test_model(model, val_classes, val_images)
