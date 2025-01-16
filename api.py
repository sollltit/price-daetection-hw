import streamlit as st
from ultralytics import YOLO
import cv2
import easyocr
import tempfile

# загрузка модели
model = YOLO('runs/detect/train3/weights/best.pt')



# функция для распознавания
def recognize_price(img, model):

    """

    Функция принимает путь к изображению, по которому надо сделать 
    предсказание и модель для детекции. 
        
    """
    # читаем изображение с помощью open-cv
    image = cv2.imread(img)
    # сохраняем результаты детекции 
    results = model(image)
    if len(results[0].boxes) != 0:
        # проходимся по результатам чтобы посмотреть координаты области с ценой
        # инициализация EasyOcr; выбираем русский и английский язык
        reader = easyocr.Reader(['en', 'ru'])
        for res in results:
            boxes = res.boxes.xyxy
            for box in boxes:
                # координаты
                x1, y1, x2, y2 = map(int, box)
    
                # обрезаем область с тестом
                crop_img = image[y1:y2, x1:x2]
                # к области примением EasyOcr
                ocr_res = reader.readtext(crop_img, allowlist='0123456789.', adjust_contrast=True)
                # вывод результата; на каждую найденную область выводиться будет 
                # предсказание, у которого выше вероятность
                max_proba = 0
                id = 0
                max_id = 0
                for detection in ocr_res:
                    proba = detection[2]
                    if proba > max_proba:
                        max_proba = proba
                        max_id = id
                    id+=1
                # вывод цены и вероятности
                for detection in [ocr_res[max_id]]:
                    text = detection[1]
                    proba = detection[2]
                    
                    # print(f'Price: {text}\nProba: {proba}')
                    # print('-'*120)
                # настройка отображения результата (фотография с рамкой детекции и подписанной ценой)
                cv2.rectangle(image, (x1, y1), (x2, y2), (143, 0, 205), 2)
                cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (143, 0, 205), 2)
            
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
    
    res = recognize_price(tmp_file_path, model) 
    st.image(res, caption='Результат')