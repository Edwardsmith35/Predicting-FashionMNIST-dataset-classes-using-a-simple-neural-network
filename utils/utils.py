

def check_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

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
