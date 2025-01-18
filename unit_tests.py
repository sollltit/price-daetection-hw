import unittest
from ultralytics import YOLO
import cv2
import easyocr

# инициализация EasyOcr; выбираем русский и английский языки

reader = easyocr.Reader(['en', 'ru'])

# загрузка модели
model_unit = YOLO('runs/detect/train3/weights/best.pt')


# функция для распознавания
def recognize_price_unittest(img, model):

    """
    
    Функция принимает путь к изображению, по которому надо сделать 
    предсказание и путь к модели для детекции. 
        
    """
    # проверка расширения файла
    if not img.lower().endswith('.jpg'):
        return False
    else:
        # читаем изображение с помощью open-cv
        image = cv2.imread(img)
        # сохраняем результаты детекции 
        results = model(image)
        
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
                    # proba = ocr_res[0][2]
                    # print(f'Цена: {text}\nВероятность правильного распознования: {proba}')
                    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 176, 65), 2)
                    # cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 137), 5)
    
        # cv2.imshow('Price', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return text


class TestRecognizePr(unittest.TestCase):
    # проверка результата
    def test_check_res(self):
        input_model = model_unit
        res = recognize_price_unittest('valid\images\original_magnit_1105_v4_jpg.rf.c9248a21f9bc11bff63bd458ef136d3d.jpg', input_model)
        self.assertEqual(res, '109')
    

    # проверка изображения
    def test_true_jpg(self):
        input_model = model_unit
        # путь к корректному изображению .jpg
        val_img_path = 'преза.pptx'
        self.assertTrue(recognize_price_unittest(val_img_path, input_model), 'Файл должен быть с расширением .jpg')
    

if __name__ == '__main__':
    unittest.main()