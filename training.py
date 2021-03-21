import copy
from config import EPOCHS, LEARNING_RATE
from torch import nn, optim
from validation import validate
from dataloaders import validation_dataloader, training_dataloader
import matplotlib.pyplot as plt


def train(model, device="cpu"):
    accuracies = []
    cnn = model.to(device)
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    max_accuracy = 0

    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(training_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()

        accuracy = float(validate(model=cnn, data=validation_dataloader))
        accuracies.append(accuracy)

        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print(f'Saving best model with accuracy: {max_accuracy}')

        print(f'Epoch: {epoch + 1}, Accuracy: {accuracy}%')

    plt.plot(accuracies)
    plt.show()
    return best_model
