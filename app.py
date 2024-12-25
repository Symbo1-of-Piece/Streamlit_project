# app.py
import streamlit as st
from model import ImageCaptioningModel
from utils import load_image

def main():
    st.set_page_config(page_title="Image Captioning App", layout="wide")
    st.title("🖼️ Генерация описаний для изображений")

    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = load_image(uploaded_file)
        
        # Отображаем загруженное изображение
        st.image(image, caption="Загруженное изображение", use_container_width=True)
        
        # Генерация описания
        st.markdown("### Прогнозируем описание изображения...")
        captions = captioning_model.predict_step([uploaded_file])
        for caption in captions:
            st.write(f"**Описание**: {caption}")

if __name__ == "__main__":
    captioning_model = ImageCaptioningModel()
    main()
