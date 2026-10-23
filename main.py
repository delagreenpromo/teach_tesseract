import os
import time
import signal
import sys
from dotenv import load_dotenv

from api_client import APIClient
from logger import logger
from ocr_processor import OCRProcessor
from pdf_processors import PdfProcessingFactory, TokenStorage
import config_env as cfg

load_dotenv()

# ----------------------- Валидация -----------------------
cfg.validate_runtime_config()
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

# ----------------------- Сборка URL -----------------------
objects_url = f"{cfg.API_BASE_URL}{cfg.API_GET_OBJECTS_ID}"
authenticate_url_template = f"{cfg.API_BASE_URL}{cfg.API_AUTH_ENDPOINT}"
blob_url_template = f"{cfg.API_BASE_URL}{cfg.API_GET_BLOB_ID}"
download_url_template = f"{cfg.API_BASE_URL}{cfg.API_DOWNLOAD_ENDPOINT}"
send_url_template = f"{cfg.API_BASE_URL}{cfg.API_SEND_CONTENT}"

# ----------------------- Единое хранилище токенов -----------------------
token_storage = TokenStorage(db_path=cfg.TOKEN_DB_PATH)

# ----------------------- Единый OCR-движок -----------------------
# Тот же класс, что и в admin_app.py (через PdfProcessingFactory → OCRPdfProcessor
# → OCRProcessor). Настраиваем из env — единый источник истины.
ocr_engine = OCRProcessor(
    dpi=cfg.OCR_DPI,
    lang=cfg.OCR_LANG,
    psm=cfg.OCR_PSM,
    min_area_ratio=cfg.OCR_MIN_AREA_RATIO,
    width_tolerance=cfg.OCR_WIDTH_TOLERANCE,
    save_images=cfg.SAVE_IMAGES,
    save_dir=cfg.RESULTS_DIR,
    use_advanced=cfg.OCR_USE_ADVANCED_RECOGNITION,
    kernel_width=cfg.OCR_KERNEL_WIDTH,
    kernel_height=cfg.OCR_KERNEL_HEIGHT,
    dilation_iterations=cfg.OCR_DILATION_ITERATIONS,
    debug=cfg.DEBUG,
    production_mode=cfg.IS_PRODUCTION,
    char_whitelist=cfg.OCR_CHAR_WHITELIST,
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
    username=cfg.API_USERNAME,
    password=cfg.API_PASSWORD,
    ocr_processor=ProcessorAdapter(token_storage, ocr_engine),
    first_attribute_id=cfg.API_FIRST_ATTRIBUTE_ID,
    last_attribute_id=cfg.API_LAST_ATTRIBUTE_ID,
    debug=cfg.DEBUG,
    is_production=cfg.IS_PRODUCTION,
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
                client.get_blob_id(blob_url_template, extensions=cfg.EXTENSIONS)
                max_files = cfg.MAX_FILES_PER_CYCLE if cfg.MAX_FILES_PER_CYCLE > 0 else None
                client.process_all_files(
                    download_url_template=download_url_template,
                    send_url_template=send_url_template,
                    selection_keywords=cfg.SELECTION_KEYWORDS,
                    exclude_keywords=cfg.EXCLUDE_KEYWORDS,
                    max_files=max_files,
                )
        except KeyboardInterrupt:
            logger.info("Ручная остановка сервиса")
            break
        except Exception as e:
            logger.exception(f"Критическая ошибка в цикле: {e}")
            time.sleep(10)
            continue

        logger.info(f"Ожидание {cfg.API_POLLING_INTERVAL} секунд до следующего опроса...")
        time.sleep(cfg.API_POLLING_INTERVAL)