import torch


def create_squeezenet_model():
    return torch.hub.load('pytorch/vision:v0.9.0', 'squeezenet1_0', pretrained=True)