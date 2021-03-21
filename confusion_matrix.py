import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from dataloaders import validation_dataloader
import matplotlib.pyplot as plt


def get_validated_predictions(model, data):
    predicted = []
    actual = []

    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        predicted.extend(list(pred.numpy()))
        actual.extend(list(labels.numpy()))

    return np.array(predicted), np.array(actual)


def generate_conf_matrix(model):
    predicted, actual = get_validated_predictions(model, validation_dataloader)

    labels = np.arange(0, 10)
    conf_matrix = pd.DataFrame(confusion_matrix(actual, predicted, labels=labels))
    cell_text = []
    for row in range(len(conf_matrix)):
        cell_text.append(conf_matrix.iloc[row])

    plt.table(cellText=cell_text, colLabels=conf_matrix.columns, rowLabels=labels, loc='center')
    plt.axis('off')
    plt.show()
