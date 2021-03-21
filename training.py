import copy
from config import EPOCHS, LEARNING_RATE
from torch import nn, optim
from validation import validate, validate_cel
from dataloaders import validation_dataloader, training_dataloader
from save_to_file import save_to_file
import matplotlib.pyplot as plt
from get_device import get_device


def train(model):
    device = get_device()
    cnn = model.to(device)

    accuracies = []
    training_losses = []
    validation_losses = []

    cel = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    max_accuracy = 0

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

        accuracy = float(validate(cnn, validation_dataloader))
        accuracies.append(accuracy)

        training_loss = float(sum(losses) / len(losses))
        training_losses.append(training_loss)

        validation_loss = float(validate_cel(cnn, validation_dataloader, cel))
        validation_losses.append(validation_loss)

        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy

            save_to_file(model)
            print(f'Saving best model with accuracy: {max_accuracy}')

        print(f'Epoch: {epoch + 1}, Accuracy: {accuracy}%')

    plt.plot(training_losses, label='Training losses')
    plt.plot(validation_losses, label='Validation losses')
    plt.plot(accuracies, label='Accuracies')
    plt.legend()
    plt.show()

    return best_model
