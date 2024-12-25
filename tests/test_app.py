import pytest
import sys
import os

# Добавляем родительскую директорию в пути поиска модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import ImageCaptioningModel
from config import MODEL_PATH, MAX_LENGTH, NUM_BEAMS
import torch
from PIL import Image

# Создаём фикстуру для модели, чтобы её можно было использовать в тестах
@pytest.fixture
def model():
    return ImageCaptioningModel()

# Тест: проверка, что модель загружается корректно
def test_model_load(model):
    assert model is not None, "Модель не загрузилась"

# Тест: проверка работы предсказания для изображения
def test_image_captioning(model):
    # Путь к тестовому изображению
    test_image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
    
    # Убедимся, что файл существует
    assert os.path.exists(test_image_path), "Тестовое изображение не найдено"
    
    # Прогоняем изображение через модель
    predictions = model.predict_step([test_image_path])
    
    # Проверяем, что предсказания не пустые
    assert len(predictions) > 0, "Предсказания пустые"

# Тест: проверка, что модель использует правильное устройство (GPU или CPU)
def test_device_selection(model):
    # Ожидаем, что модель будет работать либо на GPU, либо на CPU
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert model.device.type == expected_device, f"Модель работает не на том устройстве. Ожидаем: {expected_device}, но получаем: {model.device.type}"

# Тест: проверка, что преобразование изображения выполняется корректно
def test_image_processing(model):
    test_image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
    
    assert os.path.exists(test_image_path), "Тестовое изображение не найдено"
    
    image = Image.open(test_image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    images = [image]
    pixel_values = model.feature_extractor(images=images, return_tensors="pt").pixel_values
    assert pixel_values is not None, "Преобразование изображения не выполнено"
