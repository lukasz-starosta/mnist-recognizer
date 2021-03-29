from torch import nn


def create_own_model():
    model = nn.Sequential(
        # 1st layer - convolutional
        nn.Conv2d(1, 6, 5, padding=2),
        # ReLU outputs the input if positive, 0 otherwise
        nn.ReLU(),

        # 2nd layer - subsampling/pooling
        nn.AvgPool2d(2),

        # 3rd layer - convolutional
        nn.Conv2d(6, 16, 5, padding=0),
        nn.ReLU(),

        # 4th layer - subsampling/pooling
        nn.AvgPool2d(2),

        # Flatten to tensor
        nn.Flatten(),

        # 5th layer - fully connected
        # Applies a linear transformation of data, 2nd input * kernel size * kernel size (last conv layer)
        # outputs 120 features
        nn.Linear(16 * 5 * 5, 120),
        nn.ReLU(),

        # 6th layer - fully connected
        nn.Linear(120, 84),
        nn.ReLU(),

        # Final layer outputting the 10 digits
        nn.Linear(84, 10)
    )
    return model
