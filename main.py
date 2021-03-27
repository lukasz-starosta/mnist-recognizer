import sys
from models.squeezenet_model import create_squeezenet_model
from models.own_model import create_own_model
from utils.training import train

if len(sys.argv) > 1 and sys.argv[1] == '-s':
    print("Using Squeezenet model")
    model = create_squeezenet_model()
else:
    print("Using own model")
    model = create_own_model()

model = train(model)
