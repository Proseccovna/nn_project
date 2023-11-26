import torch
from models.model_1.model_1 import model, DEVICE
from models.model_1.preprocessing_1 import make_preproc

def predict_1(image):

    dict_classes = {
        0: 'Dew',
        1: 'Fogsmog',
        2: 'Frost',
        3: 'Glaze',
        4: 'Hail',
        5: 'Lightning',
        6: 'Rain',
        7: 'Rainbow',
        8: 'Rime',
        9: 'Sandstorm',
        10: 'Snow'
    }

    image = make_preproc(image)

    model.eval()
    predict = torch.argmax(model(image.unsqueeze(0).to(DEVICE)), dim=1).item()

    return f'Модель предсказала: {dict_classes[predict]}.'