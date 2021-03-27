import sys
from torch import save
from config import PATH_TO_OWN_MODEL, PATH_TO_SQUEEZENET_MODEL


def save_to_file(model):
    if len(sys.argv) > 1 and sys.argv[1] == '-s':
        path = PATH_TO_SQUEEZENET_MODEL
    else:
        path = PATH_TO_OWN_MODEL

    open(path, 'w+')
    save(model.state_dict(), path)
    print(f'Saved model to file: {path}')
