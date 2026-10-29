"""
Централизованная загрузка конфигурации из .env.

Заменяет прежний config.ini. Все настройки читаются из переменных
окружения один раз при импорте модуля.
"""
import os
from dotenv import load_dotenv

# Загружаем .env один раз (idempotent — повторные вызовы безопасны)
load_dotenv()


def env(key, default=None, cast=str):
    """Читает переменную окружения с приведением типа.

    Поддерживаемые cast: str (по умолчанию), bool, int, float, list.
    list — строка через запятую (',') разбивается на список с trim.
    Пустая строка трактуется как «не задано» → возвращается default.
    """
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if cast is int:
        return int(raw)
    if cast is float:
        return float(raw)
    if cast is list:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return raw


# ============================================================
# API
# ============================================================
API_BASE_URL = env("API_BASE_URL", "http://127.0.0.1:8000")
API_AUTH_ENDPOINT = env("API_AUTH_ENDPOINT", "/core/api/Auth/authenticate")
API_GET_OBJECTS_ID = env("API_GET_OBJECTS_ID", "/core/api/objects/select")
API_GET_BLOB_ID = env("API_GET_BLOB_ID", "/core/api/files/objects/{obj_id}")
API_DOWNLOAD_ENDPOINT = env("API_DOWNLOAD_ENDPOINT", "/core/api/files/objects/{obj_id}/files/{blob_id}")
API_SEND_CONTENT = env("API_SEND_CONTENT", "/core/api/objects/{obj_id}/attributes")
API_FIRST_ATTRIBUTE_ID = env("API_FIRST_ATTRIBUTE_ID", 30357, int)
API_LAST_ATTRIBUTE_ID = env("API_LAST_ATTRIBUTE_ID", 30356, int)
API_USERNAME = env("API_USERNAME", "")
API_PASSWORD = env("API_PASSWORD", "")
API_POLLING_INTERVAL = env("API_POLLING_INTERVAL", 60, int)

# ============================================================
# Paths
# ============================================================
SSL_TLS = env("SSL_TLS", True, bool)
CERT = env("CERT", None, str)

# ============================================================
# Paths
# ============================================================
RESULTS_DIR = env("RESULTS_DIR", "./results")
TOKEN_DB_PATH = env("TOKEN_DB_PATH", os.path.join(RESULTS_DIR, "unique_tokens.db"))

# ============================================================
# Processing
# ============================================================
EXTENSIONS = env("EXTENSIONS", ".dwg.pdf,.cdw.pdf", list)
IS_PRODUCTION = env("IS_PRODUCTION", True, bool)
SAVE_IMAGES = env("SAVE_IMAGES", False, bool)
DEBUG = env("DEBUG", False, bool)
MAX_FILES_PER_CYCLE = env("MAX_FILES_PER_CYCLE", 0, int)
ALLOWED_FILE_TYPES = env("ALLOWED_FILE_TYPES", "", list)

LOG_LEVEL = env("LOG_LEVEL", "INFO", str)
LOG_FILE = env("LOG_FILE", "/dev/null", str)
# LOG_MODE: console | file | both | none
LOG_MODE = env("LOG_MODE", "both", str)

# ============================================================
# OCR
# ============================================================
OCR_DPI = env("OCR_DPI", 300, int)
OCR_LANG = env("OCR_LANG", "rus")
OCR_PSM = env("OCR_PSM", 6, int)
OCR_USE_ADVANCED_RECOGNITION = env("OCR_USE_ADVANCED_RECOGNITION", False, bool)
OCR_MIN_AREA_RATIO = env("OCR_MIN_AREA_RATIO", 0.0005, float)
OCR_WIDTH_TOLERANCE = env("OCR_WIDTH_TOLERANCE", 0.3, float)
OCR_KERNEL_WIDTH = env("OCR_KERNEL_WIDTH", 20, int)
OCR_KERNEL_HEIGHT = env("OCR_KERNEL_HEIGHT", 30, int)
OCR_DILATION_ITERATIONS = env("OCR_DILATION_ITERATIONS", 3, int)
OCR_CHAR_WHITELIST = env("OCR_CHAR_WHITELIST", "")

# ============================================================
# Selection / Exclusion keywords
# ============================================================
SELECTION_KEYWORDS = env("SELECTION_KEYWORDS", "", list)
EXCLUDE_KEYWORDS = env("EXCLUDE_KEYWORDS", "", list)

# ============================================================
# Admin panel
# ============================================================
ADMIN_LOGIN = env("ADMIN_LOGIN", "admin")
ADMIN_PASSWORD = env("ADMIN_PASSWORD", "admin")
ADMIN_PORT = env("ADMIN_PORT", 5002, int)
ADMIN_SECRET_KEY = env("ADMIN_SECRET_KEY", "teach-tesseract-admin-secret")


# ============================================================
# Валидация конфигурации
# ============================================================
def validate_runtime_config():
    """Fail-fast проверки. Вызывается из main.py при старте сервиса."""
    if not IS_PRODUCTION and not DEBUG:
        raise RuntimeError(
            "Недопустимая конфигурация: оба флага IS_PRODUCTION и DEBUG "
            "установлены в False. Обработка не будет выполнять ни отправку, "
            "ни логирование."
        )

# ============================================================
# OCR: per-format настройки (автоопределение по размеру страницы)
# ============================================================
# Если формат страницы определён, OCRProcessor берёт kernel_width /
# kernel_height / dilation_iterations из этой секции.
# Если для формата ничего не задано — fallback на общие OCR_KERNEL_*.
#
# Размеры форматов (landscape, мм, длинная сторона):
#   A4: 297   A3: 420   A2: 594   A1: 841   A0: 1189
#
# Автоопределение считается как max(width, height) / DPI * 25.4 (в мм).
OCR_AUTO_FORMAT = env("OCR_AUTO_FORMAT", True, bool)

OCR_FORMAT_SETTINGS = {
    "A4": {
        "kernel_width": env("OCR_KERNEL_A4_WIDTH", 30, int),
        "kernel_height": env("OCR_KERNEL_A4_HEIGHT", 40, int),
        "dilation_iterations": env("OCR_DILATION_A4", 2, int),
    },
    "A3": {
        "kernel_width": env("OCR_KERNEL_A3_WIDTH", 30, int),
        "kernel_height": env("OCR_KERNEL_A3_HEIGHT", 60, int),
        "dilation_iterations": env("OCR_DILATION_A3", 2, int),
    },
    "A2": {
        "kernel_width": env("OCR_KERNEL_A2_WIDTH", 30, int),
        "kernel_height": env("OCR_KERNEL_A2_HEIGHT", 90, int),
        "dilation_iterations": env("OCR_DILATION_A2", 2, int),
    },
    "A1": {
        "kernel_width": env("OCR_KERNEL_A1_WIDTH", 40, int),
        "kernel_height": env("OCR_KERNEL_A1_HEIGHT", 120, int),
        "dilation_iterations": env("OCR_DILATION_A1", 2, int),
    },
    "A0": {
        "kernel_width": env("OCR_KERNEL_A0_WIDTH", 40, int),
        "kernel_height": env("OCR_KERNEL_A0_HEIGHT", 150, int),
        "dilation_iterations": env("OCR_DILATION_A0", 2, int),
    },
}