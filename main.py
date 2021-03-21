from squeezenet_model import create_squeezenet_model
from training import train
from get_device import get_device
from confusion_matrix import generate_conf_matrix

model = create_squeezenet_model()
model = train(model, get_device())
generate_conf_matrix(model)
