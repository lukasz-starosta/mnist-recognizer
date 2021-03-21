import torch


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


def validate_cel(model, data, cel):  # validation of cross-entropy loss
    results = []
    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        labels = labels.cuda()
        pred = model(images)
        results.append(cel(pred, labels))
    return sum(results) / len(results)
