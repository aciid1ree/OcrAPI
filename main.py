from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import io
import cv2
import numpy as np
from gigachat import GigaChat 
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()

api_key = os.getenv("GIGACHAT_API_KEY")

def GetPrompt(recognizedText):
    prompt = (
        "Проанализируй текст чека, распознанный с изображения. "
        "Выдели все позиции блюд и напитков с их ценами и количеством. "
        "Если названия искажены (например, из-за OCR-ошибок), восстанови их до понятного читаемого вида. "
        "Названия блюд должны быть в корректном регистре и выглядеть натурально для меню. "
        "Цены и количество должны быть указаны рядом с блюдом, игнорируя служебную информацию вроде даты, номера кассира и прочего. "
        "Если цена или количество не указаны, игнорируй эти позиции. "
        "Ответ верни строго в JSON-формате, где ключ — это строка (название блюда), а значение — массив из двух элементов: цена и количество (в рублях и в числовом формате). "
        "Также добавь ключ \"Итого\" с суммой всех цен. "
        "Ответ должен быть только валидным JSON без пояснений, примеров или лишней информации. "
        "Примерный формат: {\"Название блюда\": [цена, количество], \"Итого\": сумма}. "
        "Вот текст чека:\n"
        f"{recognizedText}"
    )

    return prompt


def extract_total_from_giga_json(json_data):
    """
    Принимает десериализованный JSON-ответ от GigaChat.
    Возвращает значение из ключа "Итого", если оно есть.
    """
    return json_data.get("Итого", None)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image: Image.Image) -> Image.Image:
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    processed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return Image.fromarray(processed)

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    processed_image = image
    print(os.getenv("API_KEY"))

    recognized_text = pytesseract.image_to_string(processed_image, lang='rus')

    with GigaChat(credentials=api_key, verify_ssl_certs=False) as giga:
        response = giga.chat(GetPrompt(recognized_text))
        reply = response.choices[0].message.content

    return {
        "recognized_text": recognized_text,
        "giga_reply": reply.strip()
    }
