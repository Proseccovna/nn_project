import torch
from torch import nn
from torchvision.models import densenet121

DEVICE = 'cpu'

model_2 = densenet121()

model_2.classifier = nn.Linear(1024, 612)
model_2.classifier_out = nn.Linear(612, 200)

model_2.load_state_dict(torch.load('bestweights_for_bird.pt', map_location=torch.device(DEVICE)))

