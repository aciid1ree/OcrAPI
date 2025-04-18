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
print(api_key) 

def GetPrompt(recognizedText):
    prompt = (
        "Проанализируй текст чека, распознанный с изображения. "
        "Выдели все позиции блюд и напитков с их ценами. "
        "Если названия искажены (например, из-за OCR-ошибок), восстанови их до понятного читаемого вида. "
        "Названия блюд должны быть в корректном регистре и выглядеть натурально для меню. "
        "Игнорируй служебную информацию вроде даты, номера кассира и прочего. "
        "Если цена указана — обязательно включи её. "
        "Ответ верни строго в JSON-формате, где ключ — это строка (название блюда), значение — число (цена в рублях). "
        "В конце добавь ключ \"Итого\" с суммой всех цен. "
        "Никаких пояснений, только валидный JSON. Пример:\n\n"
        "{\n"
        "  \"Чизбургер с картофельными чипсами\": 400,\n"
        "  \"Гозе томатный (0.5 л)\": 300,\n"
        "  \"Ржаные гренки с чесноком\": 250,\n"
        "  \"Итого\": 950\n"
        "}\n\n"
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
