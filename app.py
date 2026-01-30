import telebot
import os
import cv2
from ultralytics import YOLO

# ===== НАСТРОЙКИ =====
TOKEN = "ur token"

MODEL_PATH = r"ur path"

CLASSES = {
    0: "Деловой стиль (formal)",
    1: "Неделовой стиль (nonformal)"
}

CONF_THRESHOLD = 0.5  # порог уверенности

# ====================

bot = telebot.TeleBot(TOKEN)
model = YOLO(MODEL_PATH)

os.makedirs("temp", exist_ok=True)


@bot.message_handler(commands=["start"])
def start(message):
    bot.send_message(
        message.chat.id,
        "👔 *Детектор делового стиля*\n\n"
        "Отправь фотографию человека, и я скажу:\n"
        "— деловой это стиль или нет\n\n"
        "📸 Просто отправь фото",
        parse_mode="Markdown"
    )


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    try:
        # --- Скачивание фото ---
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        img_path = f"temp/{message.chat.id}.jpg"
        with open(img_path, "wb") as f:
            f.write(downloaded_file)

        # --- Запуск модели ---
        results = model(img_path)[0]

        if len(results.boxes) == 0:
            bot.send_message(
                message.chat.id,
                "❌ Человек не обнаружен или стиль определить не удалось"
            )
            return

        # --- Берём самый уверенный bbox ---
        best_box = max(results.boxes, key=lambda b: float(b.conf))
        cls_id = int(best_box.cls)
        confidence = float(best_box.conf)

        if confidence < CONF_THRESHOLD:
            bot.send_message(
                message.chat.id,
                "🤔 Не удалось уверенно определить стиль"
            )
            return

        label = CLASSES.get(cls_id, "Неизвестно")

        # --- Рисуем bounding box ---
        img = cv2.imread(img_path)
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"{label} ({confidence:.2f})")
        cv2.putText(
            img,
            f"{label} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        result_path = f"temp/result_{message.chat.id}.jpg"
        cv2.imwrite(result_path, img)

        # --- Отправка результата ---
        with open(result_path, "rb") as photo:
            bot.send_photo(
                message.chat.id,
                photo,
                caption=f"✅ *Результат анализа:*\n"
                        f"👕 Стиль: *{label}*\n"
                        f"📊 Уверенность: *{confidence:.2f}*",
                parse_mode="Markdown"
            )

    except Exception as e:
        bot.send_message(message.chat.id, "⚠️ Произошла ошибка при обработке изображения")
        print(e)


print("🤖 Бот запущен")
bot.polling(none_stop=True)

