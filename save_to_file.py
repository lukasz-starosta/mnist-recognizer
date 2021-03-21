from torch import save
from config import PATH_TO_MODEL


def save_to_file(model):
    open(PATH_TO_MODEL, 'w+')
    save(model, PATH_TO_MODEL)
    print(f'Saved model to file: {PATH_TO_MODEL}')
