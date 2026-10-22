import configparser
import os
import time
import signal
import sys
from dotenv import load_dotenv
from api_client import APIClient
from logger import logger
from ocr_processor import OCRProcessor
from pdf_processors import PdfProcessingFactory, TokenStorage

# ----------------------- Загрузка переменных окружения -----------------------
load_dotenv()

# ----------------------- Загрузка конфигурации -----------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.ini")
with open(CONFIG_PATH, encoding="utf-8") as f:
    config_content = os.path.expandvars(f.read())

config = configparser.ConfigParser(interpolation=None)
config.read_string(config_content)


def get_config(section, key, fallback=None, type=str):
    """Получить значение сначала из переменной окружения, потом из config.ini."""
    env_key = f"{section}_{key}".upper()
    env_value = os.environ.get(env_key)
    if env_value is not None:
        if type == bool:
            return env_value.lower() in ('true', '1', 'yes')
        return type(env_value)
    if type == bool:
        return config.getboolean(section, key, fallback=fallback)
    elif type == int:
        return config.getint(section, key, fallback=fallback)
    elif type == float:
        return config.getfloat(section, key, fallback=fallback)
    else:
        return config.get(section, key, fallback=fallback)


# ----------------------- Общие настройки -----------------------
EXTENSIONS_FILE = [ext.strip() for ext in get_config("Processing", "extensions", fallback=".dwg.pdf").split(",") if ext.strip()]
IS_PRODUCTION = get_config("Processing", "is_production", fallback=True, type=bool)
SAVE_IMAGES = get_config("Processing", "save_images", fallback=False, type=bool)
DEBUG = get_config("Processing", "debug", fallback=False, type=bool)

# ----------------------- Параметры API -----------------------
API_BASE_URL = get_config("API", "base_url")
API_AUTH_ENDPOINT = get_config("API", "authenticate_endpoint")
API_GET_OBJECTS_ID = get_config("API", "get_objects_id")
API_GET_BLOB_ID = get_config("API", "get_blob_id")
API_DOWNLOAD_ENDPOINT = get_config("API", "download_endpoint")
API_SEND_CONTENT = get_config("API", "send_content")
API_AUTH_USERNAME = get_config("API", "username")
API_AUTH_PASSWORD = get_config("API", "password")
API_FIRST_ATTRIBUTE_ID = get_config("API", "first_attribute_id")
API_LAST_ATTRIBUTE_ID = get_config("API", "last_attribute_id")

# ----------------------- Параметры OCR (для локального OCRProcessor) -----------------------
DPI = get_config("OCR", "dpi", fallback=300, type=int)
LANG = get_config("OCR", "lang", fallback="rus", type=str)
PSM = get_config("OCR", "psm", fallback=6, type=int)
USE_ADVANCED_RECOGNITION = get_config("OCR", "use_advanced_recognition", fallback=False, type=bool)
MIN_AREA_RATIO = get_config("OCR", "min_area_ratio", fallback=0.0005, type=float)
WIDTH_TOLERANCE = get_config("OCR", "width_tolerance", fallback=0.3, type=float)
KERNEL_WIDTH = get_config("OCR", "kernel_width", fallback=20, type=int)
KERNEL_HEIGHT = get_config("OCR", "kernel_height", fallback=30, type=int)
DILATION_ITERATIONS = get_config("OCR", "dilation_iterations", fallback=3, type=int)
CHAR_WHITELIST = get_config("OCR", "char_whitelist", fallback="", type=str)

# ----------------------- Пути -----------------------
RESULTS_DIR = get_config("Paths", "results_dir", fallback="results", type=str)

# ----------------------- Ключевые слова -----------------------
SELECTION_KEYWORDS = [kw.strip() for kw in get_config("Selection", "keywords", fallback="", type=str).split(",") if kw.strip()]
EXCLUDE_KEYWORDS = [kw.strip() for kw in get_config("Exclusion", "exclude_keywords", fallback="", type=str).split(",") if kw.strip()]

# ----------------------- Интервал опроса / лимиты -----------------------
POLLING_INTERVAL = get_config("API", "polling_interval", fallback=60, type=int)
MAX_FILES_PER_CYCLE = get_config("Processing", "max_files_per_cycle", fallback=0, type=int)

# ----------------------- Защита конфигурации -----------------------
if not IS_PRODUCTION and not DEBUG:
    raise RuntimeError(
        "Недопустимая конфигурация: оба флага is_production и debug установлены в False. "
        "Обработка не будет выполнять ни отправку, ни логирование."
    )

os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------- Сборка URL -----------------------
objects_url = f"{API_BASE_URL}{API_GET_OBJECTS_ID}"
authenticate_url_template = f"{API_BASE_URL}{API_AUTH_ENDPOINT}"
blob_url_template = f"{API_BASE_URL}{API_GET_BLOB_ID}"
download_url_template = f"{API_BASE_URL}{API_DOWNLOAD_ENDPOINT}"
send_url_template = f"{API_BASE_URL}{API_SEND_CONTENT}"

# ----------------------- Общее хранилище токенов -----------------------
token_storage = TokenStorage(db_path=os.path.join(RESULTS_DIR, "unique_tokens.db"))

# ----------------------- Единый OCR-движок для прода -----------------------
# Тот же класс, что и в admin_app.py (через PdfProcessingFactory → OCRPdfProcessor
# → OCRProcessor.build_default). Настраиваем из config.ini, чтобы совпадало с локальным
# тестом ocr_processor.__main__.
ocr_engine = OCRProcessor(
    dpi=DPI, lang=LANG, psm=PSM,
    min_area_ratio=MIN_AREA_RATIO,
    width_tolerance=WIDTH_TOLERANCE,
    save_images=SAVE_IMAGES, save_dir=RESULTS_DIR,
    use_advanced=USE_ADVANCED_RECOGNITION,
    kernel_width=KERNEL_WIDTH,
    kernel_height=KERNEL_HEIGHT,
    dilation_iterations=DILATION_ITERATIONS,
    debug=DEBUG,
    production_mode=IS_PRODUCTION,
    char_whitelist=CHAR_WHITELIST,
)


class ProcessorAdapter:
    """
    Адаптер под APIClient (ожидает .extract_text(pdf_bytes, file_name)).
    Идёт через PdfProcessingFactory — тем же путём, что и admin_app.py.
    """

    def __init__(self, storage, engine):
        self.storage = storage
        self.engine = engine

    def extract_text(self, pdf_bytes, file_name):
        processor = PdfProcessingFactory.create(
            file_name, token_storage=self.storage, ocr_engine=self.engine,
        )
        if processor is None:
            logger.warning(f"Тип файла отключён через ENV: {file_name}")
            return ""
        result = processor.process(pdf_bytes, source_name=file_name)
        if isinstance(result, dict):
            return result.get("text", "")
        return str(result or "")


client = APIClient(
    username=API_AUTH_USERNAME,
    password=API_AUTH_PASSWORD,
    ocr_processor=ProcessorAdapter(token_storage, ocr_engine),
    first_attribute_id=API_FIRST_ATTRIBUTE_ID,
    last_attribute_id=API_LAST_ATTRIBUTE_ID,
    debug=DEBUG,
    is_production=IS_PRODUCTION,   
)


# ----------------------- Основной цикл -----------------------
if __name__ == "__main__":
    logger.info("Сервис запущен")
    if not client.authenticate(authenticate_url_template):
        logger.error("Не удалось авторизоваться при старте, выход")
        sys.exit(1)

    def graceful_shutdown(signum, frame):
        logger.info("Получен сигнал завершения, выход")
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    while True:
        try:
            logger.info("--- Новый цикл опроса ---")
            ids = client.get_objects_id(objects_url)
            if not ids:
                logger.info("Нет объектов для обработки")
            else:
                client.get_blob_id(blob_url_template, extensions=EXTENSIONS_FILE)
                max_files = MAX_FILES_PER_CYCLE if MAX_FILES_PER_CYCLE > 0 else None
                client.process_all_files(
                    download_url_template=download_url_template,
                    send_url_template=send_url_template,
                    selection_keywords=SELECTION_KEYWORDS,
                    exclude_keywords=EXCLUDE_KEYWORDS,
                    max_files=max_files,
                )
        except KeyboardInterrupt:
            logger.info("Ручная остановка сервиса")
            break
        except Exception as e:
            logger.exception(f"Критическая ошибка в цикле: {e}")
            time.sleep(10)
            continue

        logger.info(f"Ожидание {POLLING_INTERVAL} секунд до следующего опроса...")
        time.sleep(POLLING_INTERVAL)