import os
import shutil
import numpy as np
from torchvision.transforms import functional

DIR_NAME = "wrong"
# replace previous directory
if os.path.exists(DIR_NAME):
    shutil.rmtree(DIR_NAME)

os.mkdir(DIR_NAME)


def save_wrong_result(id, image, predicted, actual):
    image = np.squeeze(image)
    image = image * 255
    img = functional.to_pil_image(image)
    img.save(f"{DIR_NAME}/{id}_pred_{predicted}_act_{actual}.png")
