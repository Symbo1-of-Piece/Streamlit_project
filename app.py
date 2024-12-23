import streamlit as st
from image_to_text import predict_step

def main():
    # Заголовок страницы и настройка
    st.set_page_config(page_title="Image Captioning App", layout="wide", page_icon="🖼️")
    st.title("✨ Генерация описаний для изображений ✨")
    
    # Стили
    st.markdown(
        """
        <style>
        .small-text {
            font-size: 18px !important;
            color: #3E3E3E;
            font-family: 'Arial', sans-serif;
        }
        .large-text {
            font-size: 32px !important;
            font-weight: bold;
            color: #2E2E2E;
            font-family: 'Arial', sans-serif;
        }
        .file-uploader-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 300px;
            margin-bottom: 20px;
        }
        .file-uploader {
            height: 100%;
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 20px;
            cursor: pointer;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Вводное сообщение
    st.markdown('<p class="small-text">Загрузите изображение, и мы создадим описание для него!</p>', unsafe_allow_html=True)
    
    # Инициализация session_state
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "image_description" not in st.session_state:
        st.session_state.image_description = None
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    # Контейнер для загрузки изображения
    with st.container():
        st.markdown("### Загрузите изображение", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    # Проверка изменения файла
    if uploaded_file is not None and uploaded_file != st.session_state.last_uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.last_uploaded_file = uploaded_file
        st.session_state.image_description = None  # Сбрасываем старое описание
        st.rerun()  # Перезагрузка для запуска предсказания
    
    # Разделение интерфейса на две колонки
    col1, col2 = st.columns([1, 2])
    
    if st.session_state.uploaded_file is not None:
        with col1:
            st.image(st.session_state.uploaded_file, caption="Загруженное изображение", use_container_width=True)
        
        with col2:
            if st.session_state.image_description is None:
                st.markdown("### Прогнозируем описание изображения...")
                with st.spinner("🔍 Обрабатываем изображение..."):
                    # Сохраняем изображение во временный файл
                    with open("temp_image.jpg", "wb") as f:
                        f.write(st.session_state.uploaded_file.getbuffer())
                    
                    # Получаем описание изображения
                    result = predict_step(['temp_image.jpg'])
                    st.session_state.image_description = result[0]
                    st.rerun()
            else:
                st.markdown('<p class="large-text">Описание изображения:</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="large-text">{st.session_state.image_description}</p>', unsafe_allow_html=True)

# Запуск приложения
if __name__ == "__main__":
    main()
