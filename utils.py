# utils.py
from PIL import Image

def load_image(image_path):
    """
    Загружает изображение по пути и конвертирует в RGB формат.
    """
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
