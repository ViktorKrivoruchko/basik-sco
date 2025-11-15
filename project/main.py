from ultralytics import YOLO
import cv2
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QHBoxLayout, QMessageBox, QTextEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import sys
import numpy as np
import csv
from datetime import datetime
import os

class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Распознавание товаров')
        self.setFixedSize(900, 700)

        self.layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()

        self.image_label = QLabel()
        self.image_label.setFixedSize(800, 450)
        self.layout.addWidget(self.image_label)

        self.button_img = QPushButton('Загрузить изображение')
        self.button_img.clicked.connect(self.load_image)
        self.button_layout.addWidget(self.button_img)

        self.button_cam = QPushButton('Включить камеру')
        self.button_cam.clicked.connect(self.start_camera)
        self.button_layout.addWidget(self.button_cam)

        self.button_test = QPushButton('Тест на обучающем')
        self.button_test.clicked.connect(self.test_inference)
        self.button_layout.addWidget(self.button_test)

        self.layout.addLayout(self.button_layout)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setFixedHeight(150)
        self.layout.addWidget(self.result_box)

        self.setLayout(self.layout)

        self.model = YOLO('runs/detect/custom_yolo8113/weights/best.pt')

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.confidence_threshold = 0.5

        self.log_file = self.create_log_file()

    def create_log_file(self):
        today = datetime.now().strftime("%Y-%m-%d")
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir, f"log_{today}.csv")
        if not os.path.exists(filename):
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time", "Source", "Label", "Confidence"])
        return filename

    def load_image(self):
        self.stop_camera()
        file_path, _ = QFileDialog.getOpenFileName()
        if file_path:
            image = cv2.imread(file_path)
            result_image = self.detect_objects(image, source='Image')
            self.display_image(result_image)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Ошибка", "Не удалось открыть камеру.")
            return
        self.timer.start(30)

    def stop_camera(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            result_image = self.detect_objects(frame, source='Camera')
            self.display_image(result_image)

    def test_inference(self):
        test_image = 'C:/Users/ViTaY/Desktop/SCO/project/SCO_products.v1i.yolov11/train/images/-Global-Village-6_jpeg.rf.905a989a334442a66185df9c993b7464.jpg'
        if os.path.exists(test_image):
            image = cv2.imread(test_image)
            if image is not None:
                print(f"Тестирование на {test_image}")
                result_image = self.detect_objects(image, source='Test')
                self.display_image(result_image)
            else:
                print(f"Ошибка загрузки изображения {test_image}")
        else:
            print(f"Изображение {test_image} не найдено!")

    def detect_objects(self, image, source='Unknown'):
        self.result_box.clear()
        results = self.model(image, conf=0.5, imgsz=640)[0]
        print(f"Всего детекций: {len(results.boxes)}")

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[cls]
            full_label = f"{label} ({conf:.2f})"

            color = (0, 255, 0) if conf >= self.confidence_threshold else (0, 255, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, full_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

            self.log_detection(label, conf, source)
            confidence_note = " (Низкая уверенность!)" if conf < self.confidence_threshold else ""
            self.result_box.append(f"Объект: {label}, уверенность: {conf:.2f}{confidence_note}")
            print(f"Обнаружен: {label}, уверенность: {conf:.2f}, координаты: ({x1}, {y1}, {x2}, {y2})")

        return image

    def log_detection(self, label, confidence, source):
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, source, label, f"{confidence:.2f}"])

    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_label.size(), aspectRatioMode=1))

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageWindow()
    window.show()
    sys.exit(app.exec_())