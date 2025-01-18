import streamlit as st
from ultralytics import YOLO
import cv2
import easyocr
import tempfile

# загрузка модели
model = YOLO('runs/detect/train3/weights/best.pt')
# инициализация EasyOcr; выбираем русский и английский языки
reader = easyocr.Reader(['en', 'ru'])


# функция для распознавания
def recognize_price_n(img, model):

    """
    
    Функция принимает путь к изображению, по которому надо сделать 
    предсказание и путь к модели для детекции. 
        
    """

    # читаем изображение с помощью open-cv
    image = cv2.imread(img)
    # сохраняем результаты детекции 
    results = model(image)
    
    if len(results[0].boxes) != 0:
        # проходимся по результатам чтобы посмотреть координаты области с ценой
        for res in results:
            for box in res.boxes:
                # получаем координации детектируемой области
                x1, y1, x2, y2 = map(int, box.xyxy[0])
        
                # распознанную область сохраняем в отдельную переменную
                crop = image[y1:y2, x1:x2]
                # применяем EasyOcr для распознавания текста
                ocr_res = reader.readtext(crop, allowlist='0123456789.')
    
                # если текст распознан, он добавляется на фото вместе с 
                # рамкой вокргу области с ценником
                if ocr_res:
                    text = ocr_res[0][1]
                    st.write(f'Цена товара: {text}')
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 176, 65), 2)
                    cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 137), 5)
    
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else: 
        img_no = cv2.imread('No.jpg')
        return cv2.cvtColor(img_no, cv2.COLOR_BGR2RGB)

st.title('Распознавание цены товара по фотографии🏷️')
st.markdown('────────────────────────────────────────────────────────────')
st.html("<p><tt><span style='font-size: 24px;'>Загрузите изображение для того, чтобы распознать на нём ценник</span></tt></p>")

uploaded_file = st.file_uploader('Выбрать изображение', 
                                 accept_multiple_files=False, type=['jpg'])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name
    
    res = recognize_price_n(tmp_file_path, model) 
    st.image(res, caption='Результат')