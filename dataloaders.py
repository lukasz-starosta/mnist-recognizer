import torch, torchvision
from config import BATCH_SIZE

transform = torchvision.transforms.ToTensor()

training_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=transform)
validation_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=transform)

training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)