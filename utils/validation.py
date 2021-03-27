import numpy as np
import torch

from utils.io.save_wrong_result import save_wrong_result


def validate(model, data):
    total = 0
    correct = 0

    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == labels)

    return correct * 100. / total


def validate_cel(model, data, cel):
    results = []
    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        labels = labels.cuda()
        pred = model(images)
        results.append(cel(pred, labels))
    return sum(results) / len(results)


def get_predicted_actual(model, data):
    predicted = []
    actual = []

    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()

        # Save wrong predictions to files
        for j, _ in enumerate(images):
            if pred[j] != labels[j]:
                save_wrong_result((i + j), images[j], pred[j], labels[j])

        predicted.extend(list(pred.numpy()))
        actual.extend(list(labels.numpy()))

    return np.array(predicted), np.array(actual)