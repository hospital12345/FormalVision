import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import os


class YOLODetector:
    def __init__(self, model_path='ur path', conf_threshold=0.25, iou_threshold=0.45):
        """
        Инициализация детектора YOLO

        Args:
            model_path: путь к обученной модели
            conf_threshold: порог уверенности
            iou_threshold: порог для NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = None  # Будем получать из модели
        self.colors = {
            'formal': (0, 255, 0),  # Зеленый для formal
            'nonformal': (0, 0, 255)  # Красный для nonformal
        }

        self.load_model()

    def load_model(self):
        """Загрузка модели YOLO"""
        try:

            if not os.path.exists(self.model_path):
                print(f"⚠️  Файл модели не найден: {self.model_path}")
                print("Ищу в возможных местах...")


                possible_paths = [
                    'best.pt',
                    'runs/detect/train/weights/best.pt',
                    'runs/detect/train2/weights/best.pt',
                    'yolo11n.pt',
                    'yolov8n.pt'
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        self.model_path = path
                        print(f"✅ Найдена модель: {path}")
                        break

            self.model = YOLO(self.model_path)
            print(f"✅ Модель загружена: {self.model_path}")

            # Получаем имена классов из модели
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"📊 Классы модели: {self.class_names}")
                print(f"📊 Количество классов: {len(self.class_names)}")
            else:

                self.class_names = {0: 'formal', 1: 'nonformal'}
                print("⚠️  Использую стандартные классы: ['formal', 'nonformal']")

            # Проверка доступности GPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"⚙️  Устройство: {device}")

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("⚠️  Использую стандартную модель YOLOv8...")
            self.model = YOLO('yolov8n.pt')
            self.class_names = self.model.names

    def predict_image(self, image_path):
        """
        Предсказание на одном изображении

        Args:
            image_path: путь к изображению

        Returns:
            annotated_image: изображение с bounding boxes
            results: сырые результаты
        """
        if not os.path.exists(image_path):
            print(f"❌ Файл не найден: {image_path}")
            return None, None

        print(f"\n🔍 Обработка изображения: {image_path}")


        image = cv2.imread(image_path)
        if image is None:
            print("❌ Не удалось загрузить изображение")
            return None, None

        print(f"📐 Размер изображения: {image.shape[1]}x{image.shape[0]}")

        # Предсказание
        try:
            results = self.model.predict(
                source=image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                save=False,
                show=False,
                verbose=False
            )
        except Exception as e:
            print(f"❌ Ошибка при предсказании: {e}")
            return None, None


        annotated_image = self.draw_boxes(image.copy(), results[0])

        # Вывод информации
        self.print_detection_info(results[0])

        return annotated_image, results[0]

    def draw_boxes(self, image, result):
        """
        Отрисовка bounding boxes на изображении

        Args:
            image: исходное изображение
            result: результаты детекции

        Returns:
            image: изображение с bounding boxes
        """
        if result.boxes is None or len(result.boxes) == 0:
            print("⚠️ Объекты не обнаружены")
            return image

        boxes = result.boxes.xyxy.cpu().numpy()  # координаты bbox
        confidences = result.boxes.conf.cpu().numpy()  # уверенность
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # классы

        print(f"📊 Найдено боксов: {len(boxes)}")

        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = map(int, box)

            # Получаем имя класса
            if cls_id in self.class_names:
                class_name = self.class_names[cls_id]
            else:
                class_name = f"class_{cls_id}"
                print(f"⚠️  Неизвестный класс ID: {cls_id}")

            # Выбираем цвет
            if class_name == 'formal' or cls_id == 0:
                color = self.colors['formal']
            elif class_name == 'nonformal' or cls_id == 1:
                color = self.colors['nonformal']
            else:
                color = (255, 255, 0)  # Желтый для других классов

            # Рисуем bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Создаем текст метки
            label = f"{class_name}: {conf:.2f}"

            # Вычисляем размер текста
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Рисуем фон для текста
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1
            )

            # Добавляем текст
            cv2.putText(
                image,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            # Точка в центре bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(image, (center_x, center_y), 3, (255, 255, 255), -1)

            print(f"  Box {i + 1}: {class_name} ({conf:.2%}) at [{x1},{y1},{x2},{y2}]")

        return image

    def print_detection_info(self, result):
        """Вывод информации о детекции"""
        if result.boxes is None or len(result.boxes) == 0:
            print("📊 Объекты не обнаружены")
            return

        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        print(f"\n📊 Статистика обнаружения:")
        print("-" * 40)

        formal_count = 0
        nonformal_count = 0
        other_count = 0

        for i, (conf, cls_id) in enumerate(zip(confidences, class_ids)):
            if cls_id in self.class_names:
                class_name = self.class_names[cls_id]
            else:
                class_name = f"class_{cls_id}"

            if class_name == 'formal' or cls_id == 0:
                formal_count += 1
            elif class_name == 'nonformal' or cls_id == 1:
                nonformal_count += 1
            else:
                other_count += 1

        print(f"👔 Formal: {formal_count}")
        print(f"👕 Non-formal: {nonformal_count}")
        if other_count > 0:
            print(f"🔶 Другие классы: {other_count}")

        print("-" * 40)

        if formal_count > nonformal_count:
            print("📝 ОБЩИЙ РЕЗУЛЬТАТ: FORMAL ОДЕЖДА")
        elif nonformal_count > formal_count:
            print("📝 ОБЩИЙ РЕЗУЛЬТАТ: NON-FORMAL ОДЕЖДА")
        elif formal_count == 0 and nonformal_count == 0 and other_count > 0:
            print("📝 Обнаружены другие объекты (не одежда)")
        else:
            print("📝 РАВНОЕ КОЛИЧЕСТВО ИЛИ НИЧЕГО НЕ ОБНАРУЖЕНО")

    def display_result(self, original_image, annotated_image, result):
        """
        Отображение результатов

        Args:
            original_image: исходное изображение
            annotated_image: изображение с bounding boxes
            result: результаты детекции
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # Оригинальное изображение
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Оригинальное изображение')
        axes[0].axis('off')

        # Аннотированное изображение
        axes[1].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Результат детекции')
        axes[1].axis('off')

        # Легенда
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Formal'),
            Patch(facecolor='red', label='Non-formal')
        ]
        axes[1].legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.show()

    def save_result(self, annotated_image, output_path=None):
        """Сохранение результата"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"detection_result_{timestamp}.jpg"

        cv2.imwrite(output_path, annotated_image)
        print(f"💾 Результат сохранен: {output_path}")
        return output_path


def find_trained_model():

    search_paths = [
        'best.pt',
        'runs/detect/train/weights/best.pt',
        'runs/detect/train2/weights/best.pt',
        'runs/detect/train3/weights/best.pt',
        r'ur patht',
    ]

    for path in search_paths:
        if os.path.exists(path):
            print(f"✅ Найдена обученная модель: {path}")
            return path

    print("⚠️  Обученная модель не найдена")
    print("Доступные файлы .pt в проекте:")
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith('.pt'):
                print(f"  - {os.path.join(root, file)}")

    return None


def main():

    print("🎯 YOLO Detector - Тестирование модели")
    print("=" * 50)

    # Поиск обученной модели
    model_path = find_trained_model()

    # Инициализация детектора
    if model_path:
        detector = YOLODetector(
            model_path=model_path,
            conf_threshold=0.25,
            iou_threshold=0.45
        )
    else:
        print("\n⚠️  Использую стандартную модель YOLO")
        detector = YOLODetector(
            model_path='yolov8n.pt',
            conf_threshold=0.25,
            iou_threshold=0.45
        )


    while True:
        print("\n📁 Введите путь к изображению (или 'exit' для выхода):")
        image_path = input().strip()

        if image_path.lower() == 'exit':
            print("👋 Выход...")
            break

        # Проверка существования файла
        if not os.path.exists(image_path):
            print(f"❌ Файл не найден: {image_path}")

            # Попробовать в текущей директории
            if os.path.exists(os.path.basename(image_path)):
                image_path = os.path.basename(image_path)
                print(f"✅ Найден файл в текущей директории: {image_path}")
            else:
                print("Попробуйте еще раз")
                continue

        # Обработка изображения
        original_image = cv2.imread(image_path)
        annotated_image, result = detector.predict_image(image_path)

        if annotated_image is not None:

            detector.display_result(original_image, annotated_image, result)

            # Сохранение результата
            save_option = input("\n💾 Сохранить результат? (y/n): ").lower()
            if save_option == 'y':
                output_name = f"result_{Path(image_path).stem}.jpg"
                detector.save_result(annotated_image, output_name)

            # Дополнительная информация
            print("\n📊 Подробная информация:")
            print(f"Файл: {Path(image_path).name}")
            print(f"Размер: {original_image.shape[1]}x{original_image.shape[0]}")

        # Спросить о продолжении
        cont = input("\n🔄 Продолжить с другим изображением? (y/n): ").lower()
        if cont != 'y':
            break


if __name__ == "__main__":

    from datetime import datetime


    main()
