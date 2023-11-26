import torch
from torchvision import transforms as T


func_preproc = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])


def make_preproc(image):
    return func_preproc(image)