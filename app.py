from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
import uuid
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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

model = YOLO(MODEL_PATH)

# шрифт с кириллицей (Windows можно arial.ttf)
FONT_PATH = "arial.ttf"

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

        file = request.files.get("photo")

        if not file or file.filename == "":
            return render_template("index.html", error="Файл не выбран")

        filename = f"{uuid.uuid4()}.jpg"

        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)

        results = model(img_path)[0]

        if len(results.boxes) == 0:
            return render_template("index.html", error="Человек не найден")

        best_box = max(results.boxes, key=lambda b: float(b.conf))

        cls_id = int(best_box.cls)
        confidence = float(best_box.conf)

        if confidence < CONF_THRESHOLD:
            return render_template("index.html", error="Низкая уверенность модели")

        label = CLASSES.get(cls_id, "Неизвестно")

        # =========================
        # РИСОВАНИЕ
        # =========================

        img = cv2.imread(img_path)

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        # рамка
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # PIL для текста
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype(FONT_PATH, 20)
        except:
            font = ImageFont.load_default()

        text = f"{label} ({confidence:.2f})"

        draw.text((x1, y1 - 25), text, font=font, fill=(0, 255, 0))

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # сохранение
        result_filename = f"result_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)

        cv2.imwrite(result_path, img)

        result_image = result_path.replace("\\", "/")

    return render_template(
        "index.html",
        result_image=result_image,
        label=label,
        confidence=confidence,
        error=error
    )


# =========================
# START
# =========================

if __name__ == "__main__":
    app.run(debug=True)
