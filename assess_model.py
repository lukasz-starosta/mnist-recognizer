import sys
from config import PATH_TO_OWN_MODEL, PATH_TO_SQUEEZENET_MODEL
from utils.get_device import get_device
from models.squeezenet_model import create_squeezenet_model
from models.own_model import create_own_model
import torch
from utils.dataloaders import validation_dataloader, training_dataloader
from utils.confusion_matrix import generate_conf_matrix
from utils.validation import get_predicted_actual, validate

device = get_device()

if len(sys.argv) > 1 and sys.argv[1] == '-s':
    print("Using Squeezenet model")
    model = create_squeezenet_model()
    path = PATH_TO_SQUEEZENET_MODEL
else:
    print("Using own model")
    model = create_own_model()
    path = PATH_TO_OWN_MODEL

model.to(device)

# data = training_dataloader
data = validation_dataloader

# Load from map of layers to parameter tensors
model.load_state_dict(torch.load(path))

accuracy = validate(model, data)
print(f"Accuracy: {accuracy}%")

predicted, actual = get_predicted_actual(model, data)
generate_conf_matrix(predicted, actual)
