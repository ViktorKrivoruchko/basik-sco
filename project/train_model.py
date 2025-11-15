from ultralytics import YOLO
import os

model_path = 'yolov11n.pt'
data_path = 'SCO_products.v2i.yolov11/data.yaml'

if not os.path.exists(model_path):
    print(f"Ошибка: файл {model_path} не найден!")
    exit(1)

if not os.path.exists(data_path):
    print(f"Ошибка: файл {data_path} не найден! Проверьте путь относительно {os.getcwd()}")
    exit(1)

model = YOLO(model_path)

model.train(
    data=data_path,
    epochs=200,
    imgsz=640,
    batch=16,
    name='custom_yolo811',
    augment=False,
    patience=20,
    lr0=0.0005,
    momentum=0.937,
    weight_decay=0.0005
)

print("Обучение завершено! Модель сохранена в runs/detect/custom_yolo811/weights/best.pt")