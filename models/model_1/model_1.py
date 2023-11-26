import torch
from torch import nn
from torchvision.models import resnet34

DEVICE = 'cpu'

model = resnet34()
model.fc = nn.Linear(512, 11)

model.load_state_dict(torch.load('best_params_resnet34.pt', map_location=torch.device(DEVICE)))
