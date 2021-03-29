import copy
import time
from config import EPOCHS, LEARNING_RATE
from torch import nn, optim, cuda
from utils.validation import validate, validate_cel
from utils.dataloaders import validation_dataloader, training_dataloader
from utils.io.save_to_file import save_to_file
import matplotlib.pyplot as plt
from utils.get_device import get_device


def train(model):
    time_start = time.time()

    device = get_device()
    cnn = model.to(device)

    accuracies = []
    training_losses = []
    validation_losses = []
    max_accuracy = 0

    cel = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        losses = []
        for i, (images, labels) in enumerate(training_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cel(pred, labels)
            losses.append(loss)
            loss.backward()
            optimizer.step()

        training_loss = float(sum(losses) / len(losses))
        training_losses.append(training_loss)

        cuda.empty_cache()

        validation_loss, accuracy = validate_cel(cnn, validation_dataloader, cel)
        validation_losses.append(validation_loss.cpu())
        accuracies.append(accuracy)

        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy

        print(
            f'Epoch: {epoch + 1}, Accuracy: {accuracy}%, Training loss: {training_loss}, Validation loss: {validation_loss}')

    time_end = time.time()
    print(f'Training complete. Time elapsed: {time_end - time_start}s')

    print(f'Saving best model with accuracy: {max_accuracy}')
    save_to_file(best_model)

    plt.plot(accuracies, label='Accuracy')
    plt.legend()
    plt.show()

    plt.cla()
    plt.plot(training_losses, label='Training losses')
    plt.plot(validation_losses, label='Validation losses')
    plt.legend()
    plt.show()

    return best_model
