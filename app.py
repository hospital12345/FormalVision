from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
import uuid
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from werkzeug.utils import secure_filename

app = Flask(__name__)

# =========================
# НАСТРОЙКИ
# =========================

MODEL_PATH = r"best.pt"

CLASSES = {
    0: "Деловой стиль (formal)",
    1: "Неделовой стиль (nonformal)"
}

CONF_THRESHOLD = 0.5

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# =========================
# ЗАЩИТА ОТ ПЕРЕГРУЗКИ
# =========================

MAX_QUEUE_WORKERS = 2
REQUEST_TIMEOUT = 20

executor = ThreadPoolExecutor(max_workers=MAX_QUEUE_WORKERS)

# =========================
# ЛИМИТ РАЗМЕРА ФАЙЛА
# =========================

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# =========================
# МОДЕЛЬ
# =========================

model = YOLO(MODEL_PATH)

FONT_PATH = "arial.ttf"

# =========================
# ПРОВЕРКА ФАЙЛА
# =========================

def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# =========================
# ОСНОВНАЯ ОБРАБОТКА
# =========================

def process_image(img_path, filename):

    results = model(img_path)[0]

    if len(results.boxes) == 0:
        raise ValueError("Человек не найден")

    best_box = max(results.boxes, key=lambda b: float(b.conf))

    cls_id = int(best_box.cls)
    confidence = float(best_box.conf)

    if confidence < CONF_THRESHOLD:
        raise ValueError("Низкая уверенность модели")

    label = CLASSES.get(cls_id, "Неизвестно")

    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Ошибка чтения изображения")

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

    # рамка
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # PIL для текста
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except Exception:
        font = ImageFont.load_default()

    text = f"{label} ({confidence:.2f})"

    draw.text(
        (x1, max(0, y1 - 25)),
        text,
        font=font,
        fill=(0, 255, 0)
    )

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    result_filename = f"result_{filename}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)

    success = cv2.imwrite(result_path, img)

    if not success:
        raise RuntimeError("Ошибка сохранения результата")

    return {
        "result_image": result_path.replace("\\", "/"),
        "label": label,
        "confidence": confidence
    }

# =========================
# ГЛАВНАЯ СТРАНИЦА
# =========================

@app.route("/", methods=["GET", "POST"])
def index():

    result_image = None
    label = None
    confidence = None
    error = None

    if request.method == "POST":

        try:

            file = request.files.get("photo")

            if not file or file.filename == "":
                raise ValueError("Файл не выбран")

            if not allowed_file(file.filename):
                raise ValueError("Недопустимый формат файла")

            original_name = secure_filename(file.filename)
            ext = original_name.rsplit(".", 1)[1].lower()

            filename = f"{uuid.uuid4()}.{ext}"

            img_path = os.path.join(UPLOAD_FOLDER, filename)

            file.save(img_path)

            # =========================
            # ОЧЕРЕДЬ ЗАПРОСОВ
            # =========================

            future = executor.submit(
                process_image,
                img_path,
                filename
            )

            data = future.result(timeout=REQUEST_TIMEOUT)

            result_image = data["result_image"]
            label = data["label"]
            confidence = data["confidence"]

        except TimeoutError:
            error = "Сервер перегружен. Попробуйте позже."

        except ValueError as e:
            error = str(e)

        except Exception as e:
            print("ERROR:", e)
            error = "Внутренняя ошибка сервера"

    return render_template(
        "index.html",
        result_image=result_image,
        label=label,
        confidence=confidence,
        error=error
    )

# =========================
# ОБРАБОТКА 413
# =========================

@app.errorhandler(413)
def too_large(e):
    return render_template(
        "index.html",
        error="Файл слишком большой (макс. 10MB)"
    )

# =========================
# START
# =========================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )
