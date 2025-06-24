import torch 
import matplotlib.pyplot as plt

# Test the Model 2
def test_model(model, classes, val_images):
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