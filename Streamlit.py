import torch
import streamlit as st
import io
import imageio
import requests
from PIL import Image
from torchvision import transforms as T
from torchvision.models import densenet121
# from torchvision import io
import torch.nn as nn
import time
from models.model_1.predict_model_1 import predict_1
from models.model_2.predict_model_2 import predict_2

# st.set_page_config(layout='wide')

st.title('Проект • Введение в нейронные сети')


with st.sidebar:
    st.header('Выберите страницу')
    page = st.selectbox("Выберите страницу", ["Главная", "Времена года", "Кстати, о птичках", "Итоги"])

if page == "Главная":
    st.header('Выполнила команда "DenseNet":')
    st.subheader('🐱Руслан')
    st.subheader('🐱Тата')

    st.header(" 🌟 " * 10)

    st.header('Наши датасеты:')
    st.subheader(
        '*Задача №1*: Классификация изображений природы по временам года (обучение на основе датасета *Weather Image Recognition*)')

    st.subheader('*Задача №2*: Определение вида птички по фотографии.')


elif page == "Времена года":
    st.header("Процесс обучения:")
    st.subheader("- Модель: *resenet34*")
    st.subheader("- Оптимизатор: Adam со всеми гиперпараметрами по умолчанию.")
    st.subheader("- Обучаемые слои модели: \n- Выходной полносвязанный слой с 11-ю выходами.")

    st.info('Расширение картинки обязательно должно быть в формате .jpg/.jpeg')
    image_url = st.text_input("Введите URL изображения погоды")
    start_time = time.time()

    if image_url:
        # Загрузка изображения по ссылке
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        st.subheader('Ваше фото:')
        st.image(image)
        st.subheader('Предсказание модели:')
        st.write(f'- {predict_1(image)}')
        st.subheader(f'Время предсказания: {round((time.time() - start_time), 2)} сек.')

        st.header('🎈' * 10)


elif page == "Кстати, о птичках":
    st.header("Процесс обучения:")
    st.subheader("- Модель: *densenet121*")
    st.subheader("- Оптимизатор: Adam со всеми гиперпараметрами по умолчанию.")
    st.subheader("- Обучаемые слои модели: \n- *classifier* с входными 1024 и выходными 612 сигналами. \
    \n- *classifier_out*  с входными 612 и выходными 200 сигналами")

    st.info('Расширение картинки обязательно должно быть в формате .jpg/.jpeg')
    image_url2 = st.text_input("Введите URL изображения вашей птички")
    start_time2 = time.time()

    if image_url2:
        # Загрузка изображения по ссылке
        response2 = requests.get(image_url2)
        image2 = Image.open(io.BytesIO(response2.content))
        st.subheader('Фото вашей птички:')
        st.image(image2)
        st.subheader("Мы узнали, что это за птичка:")
        st.write(f'- {predict_2(image2)}')
        st.subheader(f'Время предсказания: {round((time.time() - start_time2), 2)} сек.')

        st.header('🎈' * 10)



elif page == "Итоги":
    st.header('Результаты и выводы')
    st.subheader('Задача №1')

    st.subheader("Точность предсказания на тренировочной и валидационной выборках (max = 0.75)")
    image6 = imageio.imread('acc1.png')[:, :, :]
    st.image(image6, caption="Caption")

    st.subheader("Loss на тренировочной и валидационной выборках")
    image7 = imageio.imread('loss1.png')[:, :, :]
    st.image(image7, caption="Caption")

    st.subheader('Задача №2')
    st.subheader("Точность предсказания на тренировочной и валидационной выборках (max = 0.65)")

    image3 = imageio.imread('Acc2.jpg')[:, :, :]
    st.image(image3, caption="Caption")

    st.subheader("Loss на тренировочной и валидационной выборках")

    image4 = imageio.imread('loss2.jpg')[:, :, :]
    st.image(image4, caption="Caption")

    st.subheader('Так же мы пробовали: Модель resnet101')
    image5 = imageio.imread('101.jpg')[:, :, :]
    st.image(image5, caption="Caption")
    st.write('> **В процессе обучения на данных о различных видах птиц, очень быстро переобучилась, \
                после чего пришлось остановить обучение. Связано это с тем, что модель не имеет ни \
                одного Dropout слоя, а кол-во слоёв более 300. В выходном уровне классификации, \
                был добавлен ещё один Fully Connecter слой, с 1028 входами и 200-ми выходами (количеством классов птиц).**')

    # РАССКАЗ О ТОМ, КАК НАМ БЫЛО ТЯЖЕЛО, НО МЫ СПРАВИЛИСЬ