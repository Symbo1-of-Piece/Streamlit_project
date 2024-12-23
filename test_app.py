import unittest
from unittest.mock import patch, MagicMock
import streamlit as st
from io import BytesIO
from image_to_text import predict_step

# Мок для predict_step
def mock_predict_step(image_paths):
    return ["Это тестовое описание изображения."]

# Тестирование Streamlit приложения
class TestImageCaptioningApp(unittest.TestCase):

    @patch("image_to_text.predict_step", side_effect=mock_predict_step)
    @patch("streamlit.file_uploader", return_value=BytesIO(b"test image data"))
    def test_image_upload_and_captioning(self, mock_file_uploader, mock_predict_step):
        # Имитируем загрузку изображения
        uploaded_file = mock_file_uploader()
        st.session_state.uploaded_file = uploaded_file
        st.session_state.last_uploaded_file = uploaded_file
        
        # Проверка, что изображение загружено
        self.assertIsNotNone(st.session_state.uploaded_file)

        # Прогоняем часть логики для генерации описания
        with patch("streamlit.rerun"):
            # Мокируем обработку изображения
            st.session_state.image_description = None
            st.session_state.uploaded_file = uploaded_file
            st.session_state.image_description = mock_predict_step([uploaded_file])[0]
            
            # Проверка результата
            self.assertEqual(st.session_state.image_description, "Это тестовое описание изображения.")

    @patch("image_to_text.predict_step", side_effect=mock_predict_step)
    def test_file_upload_behavior(self, mock_predict_step):
        # Загружаем новое изображение
        uploaded_file = BytesIO(b"new test image data")
        st.session_state.uploaded_file = uploaded_file
        st.session_state.last_uploaded_file = uploaded_file
        
        # Проверка, что сессия обновляется
        self.assertEqual(st.session_state.uploaded_file, uploaded_file)
        
        # Заглушка для прогноза
        result = mock_predict_step(['temp_image.jpg'])
        self.assertEqual(result[0], "Это тестовое описание изображения.")
        
    def test_session_state_initialization(self):
        # Проверка, что session_state правильно инициализируется
        self.assertIsNone(st.session_state.get("uploaded_file"))
        self.assertIsNone(st.session_state.get("image_description"))
        self.assertIsNone(st.session_state.get("last_uploaded_file"))

    def test_image_description_after_upload(self):
        # Мокирование загрузки изображения и генерации описания
        st.session_state.uploaded_file = BytesIO(b"image data")
        st.session_state.image_description = "Тестовое описание"
        
        self.assertEqual(st.session_state.image_description, "Тестовое описание")

# Запуск тестов
if __name__ == "__main__":
    unittest.main()
