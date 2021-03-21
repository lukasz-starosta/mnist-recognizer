import sys
from squeezenet_model import create_squeezenet_model
from own_model import create_own_model
from training import train
from confusion_matrix import generate_conf_matrix

if len(sys.argv) > 1 and sys.argv[1] == '-s':
    print("Using Squeezenet model")
    model = create_squeezenet_model()
else:
    print("Using own model")
    model = create_own_model()

model = train(model)
generate_conf_matrix(model)
